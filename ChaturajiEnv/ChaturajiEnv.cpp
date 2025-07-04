#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <bitset>
#include <cassert>
#include <vector>
#include <mutex>
#include <deque>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <random>
#include <zip.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/version.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcountll __popcnt64
#endif

// Macro for debug assertions
#define D_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed: " << msg << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace py = pybind11;
using uint = unsigned int;

std::mt19937& global_rng() {
    static thread_local std::mt19937 gen([]() {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq seq{ std::random_device{}(), static_cast<unsigned>(seed) };
        return std::mt19937(seq);
        }());
    return gen;
}

constexpr const char* notation_x[4][8] = {
    { "d", "e", "f", "g", "h", "i", "j", "k" },    // red
    { "11", "10", "9", "8", "7", "6", "5", "4" },  // blue
    { "k", "j", "i", "h", "g", "f", "e", "d" },    // yellow
    { "4", "5", "6", "7", "8", "9", "10", "11" }   // green
};

constexpr const char* notation_y[4][8] = {
    { "4", "5", "6", "7", "8", "9", "10", "11" },   // red
    { "d", "e", "f", "g", "h", "i", "j", "k" },    // blue
    { "11", "10", "9", "8", "7", "6", "5", "4" },  // yellow
    { "k", "j", "i", "h", "g", "f", "e", "d" },    // green
};

// Named constants for colors
constexpr uint RED = 0;
constexpr uint YELLOW = 2;
constexpr uint GREEN = 3;
constexpr uint BLUE = 1;

struct position {
    uint x;
    uint y;

    friend std::ostream& operator<<(std::ostream& os, const position& pos) {
        os << "(" << pos.x << ", " << pos.y << ")";
        return os;
    }

    bool operator==(const position& other) const {
        return x == other.x && y == other.y;
    }

    std::size_t hash() const {
        return std::hash<int>()(x) ^ (std::hash<int>()(y) << 1);
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& x;
        ar& y;
    }
};

bool isValid(position& p) {
    if (p.x < 0 || p.x > 7) {
        return false;
    }
    if (p.y < 0 || p.y > 7) {
        return false;
    }
    return true;
}

bool isValid(uint x, uint y) {
    if (x < 0 || x > 7) {
        return false;
    }
    if (y < 0 || y > 7) {
        return false;
    }
    return true;
}

// Convert 2D (x, y) to 1D bit index
constexpr uint index(uint x, uint y, uint color = RED) {
    D_ASSERT(x < 8 && y < 8, "wrong index");
    switch (color) {
    case RED: return y * 8 + x;
    case BLUE: return (7 - y) * 8 + x;
    case YELLOW: return (7 - x) * 8 + (7 - y);
    case GREEN: return y * 8 + (7 - x);
    default: D_ASSERT(false, "wrong color");
    }
}

// Helper to convert a uint64_t board into the flat array
uint fill_layer(uint64_t board, std::vector<float>& flattened, uint layer_index) {
    for (int i = 0; i < 64; ++i) {
        flattened[layer_index++] = (board & (1ULL << i)) ? 1.0f : 0.0f;
    }
    return layer_index;
}

struct move {
    position from;
    position to;
    uint reward = 0;

    bool operator==(const move& m) const {
        return std::tie(from.x, from.y, to.x, to.y) == std::tie(m.from.x, m.from.y, m.to.x, m.to.y);
    }

    bool operator<(const move& m) const {
        return std::tie(from.x, from.y, to.x, to.y) < std::tie(m.from.x, m.from.y, m.to.x, m.to.y);
    }

    friend std::ostream& operator<<(std::ostream& os, const move& m) {
        os << m.from << " -> " << m.to << " with reward: " << m.reward;
        return os;
    }

    //default constructor //TODO: remove?
    move() = default;

    //define constructor
    move(position from, position to, uint reward = 0) : from(from), to(to), reward(reward) {}

    //returns index on flatten action_mask of size 64*64
    uint64_t getIndex() const {
        return ((from.y << 3) | from.x) << 6 | ((to.y << 3) | to.x);
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& from;
        ar& to;
        ar& reward;
    }
};

struct move_hash {
    std::size_t operator()(const move& m) const {
        std::size_t h1 = m.from.hash();
        std::size_t h2 = m.to.hash();
        return m.from.hash() ^ (m.to.hash() << 1);
    }
};

struct Layer {
    uint64_t board; // 8x8 chessboard represented as 64 bits

    Layer() : board(0) {} // Initialize all bits to 0

    Layer(uint64_t board) : board(board) {} // Initialize with a given board

    // Set a bit at (x, y) to the given value
    void set(uint x, uint y, bool value, uint color = RED) {
        D_ASSERT(x < 8 && y < 8, "wrong index");
        uint bit = index(x, y, color);
        if (value)
            board |= (1ULL << bit); // Set the bit
        else
            board &= ~(1ULL << bit); // Clear the bit
    }

    // Get the value of the bit at (x, y)
    bool get(uint x, uint y, uint color = RED) const {
        int bit = index(x, y, color);
        return (board >> bit) & 1ULL;
    }

    uint64_t getBoard() const {
        return board;
    }

    //clockwise rotation
    Layer rotate_to_general(uint n = RED) const {
        Layer rotated;
        if (n == RED) {
            return *this;
        }
        if (n == BLUE) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    rotated.set(y, 7 - x, get(x, y));
                }
            }
        }
        if (n == YELLOW) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    rotated.set(7 - x, 7 - y, get(x, y));
                }
            }
        }
        if (n == GREEN) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    rotated.set(7 - y, x, get(x, y));
                }
            }
        }
        return rotated;
    }

    // anticlockwise rotation
    Layer rotate_to_local(uint n = RED) const {
        return rotate_to_general((4 - n) % 4);
    }

    // Logical AND with another ChessBoard
    Layer operator&(const Layer& other) const {
        Layer result;
        result.board = board & other.board;
        return result;
    }

    // Logical OR with another ChessBoard
    Layer operator|(const Layer& other) const {
        Layer result;
        result.board = board | other.board;
        return result;
    }

    uint countBits() const {
        return (uint) __builtin_popcountll(board); // GCC/Clang
    }

    // Print the board for debugging
    void print() const {
        for (int y = 7; y >= 0; --y) { // Print from top to bottom
            for (int x = 0; x < 8; ++x) {
                std::cout << (get(x, y) ? '1' : '0') << " ";
            }
            std::cout << "\n";
        }
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& board;
    }
};

Layer pawn_map() { //3840
    Layer pawn;
    pawn.set(0, 1, true);
    pawn.set(1, 1, true);
    pawn.set(2, 1, true);
    pawn.set(3, 1, true);
    return pawn;
}

Layer rook_map() { //1
    Layer rook;
    rook.set(0, 0, true);
    return rook;
}

Layer knight_map() { //2
    Layer knight;
    knight.set(1, 0, true);
    return knight;
}

Layer bishop_map() { //4
    Layer bishop;
    bishop.set(2, 0, true);
    return bishop;
}

Layer king_map() { //8
    Layer king;
    king.set(3, 0, true);
    return king;
}

struct player {
    Layer _pawn;
    Layer _rook;
    Layer _knight;
    Layer _bishop;
    Layer _king;
    Layer _attacking; //not used
    uint score;
    uint color;
    bool active;

    //define default constructor
    player(uint c = RED) {
        _pawn = Layer(3840).rotate_to_general(c);
        _rook = Layer(1).rotate_to_general(c);
        _knight = Layer(2).rotate_to_general(c);
        _bishop = Layer(4).rotate_to_general(c);
        _king = Layer(8).rotate_to_general(c);
        _attacking = Layer();
        score = 0;
        color = c;
        active = true;
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& _pawn;
        ar& _rook;
        ar& _knight;
        ar& _bishop;
        ar& _king;
        ar& _attacking; //not used
        ar& score;
        ar& color;
        ar& active;
    }
};

//players in array, starting from red --> not starting from player on turn! 
// player boards are represented in position of next turn player, f.e. if turn = BLUE, blue is in terms of (x , y) position as red at the beginning
//turn - index of player on turn
struct state {
    std::array<player, 4> players;
    uint turn;
    bool finished = false;
    uint no_progress_count = 0;

    //define default constructor
    state() {
        for (uint i = 0; i < 4; i++) {
            players[i] = player(i);
        }
        turn = RED;
    }

    //copy constructor
    state(const state& other) : players(other.players), turn(other.turn), finished(other.finished), no_progress_count(other.no_progress_count) {}

    void nextTurn() {
        for (uint i = 0; i < 3; i++) { 
            turn = (turn + 1) % 4;
            no_progress_count++;
            for (auto& player : players) {
                player._pawn = player._pawn.rotate_to_local(1);
                player._rook = player._rook.rotate_to_local(1);
                player._knight = player._knight.rotate_to_local(1);
                player._bishop = player._bishop.rotate_to_local(1);
                player._king = player._king.rotate_to_local(1);
                player._attacking = player._attacking.rotate_to_local(1);
            }
            if (no_progress_count >= 50) {
                finished = true;
                return;
            }
            if (players[turn].active) {
                return;
            }
        }
        finished = true;
    }

    bool isEmpty(position p) {
        for (auto& player : players) {
            if (player._pawn.get(p.x, p.y)
                || player._rook.get(p.x, p.y)
                || player._knight.get(p.x, p.y)
                || player._bishop.get(p.x, p.y)
                || player._king.get(p.x, p.y)) {
                return false;
            }
        }
        return true;
    }

    //rew 0 is player isn't active, otherwise reward of the piece
    bool isOccupied(position p, uint& rew);

    std::vector<move> getLegalMovesLocal(position& p);

    bool isLegalMove(move& m) {
        auto moves = getLegalMovesLocal(m.from);
        for (auto& move : moves) {
            if (move == m) {
                return true;
            }
        }
        return false;
    };

    void makeMove(move& m);

    std::vector<move> getLegalMoves();

    py::array_t<float> getLegalMoveMask();

    void printBoard();

    void printScore() const;

    void printTurn();

    void printLegalMoves();

    //gymnasium like interface
    std::pair<state, std::string> reset() {
        for (uint i = 0; i < 4; i++) {
            players[i] = player(i);
        }
        turn = RED;
        finished = false;
        return { state(), "" };
    }

    move sample() {
        auto moves = getLegalMoves();
        return moves[rand() % moves.size()];
    }

    //observation (new state), reward, terminated, truncated, info
    std::tuple<state, uint, bool, bool, std::string> step(move& m) {
        makeMove(m);
        return { *this, m.reward, finished, false, "" };
    }

    
    std::array<uint, 4> get_score() const {
        return { players[turn % 4].score, players[(turn + 1) % 4].score, players[(turn + 2) % 4].score, players[(turn + 3) % 4].score };
    }
    

    //returns reward at the end of the game, indexed from player on turn
    std::array<float, 4> getFinalReward() const {
        // (+1 for 1st player, +0.33 for 2nd, -0.33 for 3rd and -1 for 4th)

        std::vector<float> rank_rewards = { 1.0f, 0.33f, -0.33f, -1.0f };

        std::vector<std::pair<uint, int>> scores; // {score, player_index}
        for (int i = 0; i < 4; ++i) {
            scores.emplace_back(players[(turn + i) % 4].score, i);
        }
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
            return a.first > b.first; // Higher scores come first
        });

        // Assign rewards, considering ties
        std::vector<float> rewards(4, 0.0f);
        for (int i = 0; i < 4;) {
            // Find the range of tied players
            int j = i;
            while (j < 4 && scores[j].first == scores[i].first) {
                ++j;
            }
            float avg_reward = std::accumulate(rank_rewards.begin() + i, rank_rewards.begin() + j, 0.0f) / (j - i);
            for (int k = i; k < j; ++k) {
                rewards[scores[k].second] = avg_reward;
            }
            i = j;
        }

        return { rewards[0], rewards[1], rewards[2], rewards[3] };
    }

    
    //returns score indexed from 1st player (red)
    std::array<uint, 4> get_score_default() const {
        return { players[0].score, players[1].score, players[2].score, players[3].score };
    }
    

    // Convert the state into a flat vector of floats for NumPy
    py::array_t<float> to_numpy() const {
        std::vector<float> flattened(1280, 0.0f); // Total size: 1280 = 4 * 5 * 8 * 8

        uint layer_index = 0;
        for (int i = 0; i < 4; i++) {
            auto& player = players[(turn + i) % 4];
            layer_index = fill_layer(player._pawn.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._rook.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._knight.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._bishop.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._king.getBoard(), flattened, layer_index);
        }

        // Return as a NumPy array
        return py::array_t<float>(
            { 20, 8, 8 },              // Shape (5 chanels for each player, 8, 8)
            { 8 * 8 * sizeof(float),   // Stride for c
             8 * sizeof(float),        // Stride for w
             sizeof(float) },          // Stride for h
            flattened.data()           // Pointer to data
            );
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& players;
        ar& turn;
        ar& finished;
        ar& no_progress_count;
    }
};

//gymnasium like interface
state make() {
    return state();
}

bool state::isOccupied(position p, uint& rew) {
    for (auto& player : players) {
        if (player.color == turn) {
            continue;
        }
        if (player._pawn.get(p.x, p.y)) {
            rew = player.active ? 1 : 0;
            return true;
        }
        if (player._rook.get(p.x, p.y)) {
            rew = player.active ? 5 : 0;
            return true;
        }
        if (player._knight.get(p.x, p.y)) {
            rew = player.active ? 3 : 0;
            return true;
        }
        if (player._bishop.get(p.x, p.y)) {
            rew = player.active ? 5 : 0;
            return true;
        }
        if (player._king.get(p.x, p.y)) {
            rew = player.active ? 10 : 0;
            return true;
        }
    }
    rew = 0;
    return false;
}

//assumes position is given in local coordinates (player starts in corner (0,0) with rook)
std::vector<move> state::getLegalMovesLocal(position& p) {
    std::vector<move> legalMoves;
    auto& player = players[turn];
    if (player._pawn.get(p.x, p.y)) {
        if (p.y < 7) {
            uint rew;
            if (isEmpty({ p.x, p.y + 1 })) {
                legalMoves.emplace_back(move({ p, {p.x, p.y + 1}, 0 }));
            }
            if (p.x < 7) {
                if (isOccupied({ p.x + 1, p.y + 1 }, rew)) {
                    legalMoves.emplace_back(move({ p, {p.x + 1, p.y + 1}, rew }));
                }
            }
            if (p.x > 0) {
                if (isOccupied({ p.x - 1, p.y + 1 }, rew)) {
                    legalMoves.emplace_back(move({ p, {p.x - 1, p.y + 1}, rew }));
                }
            }
        }
    }
    else if (player._rook.get(p.x, p.y)) {
        for (uint i = 1; i < 8; i++) {
            if (p.y + i < 8) {
                uint rew;
                if (isEmpty({ p.x, p.y + i })) {
                    legalMoves.emplace_back(move({ p, {p.x, p.y + i}, 0 }));
                }
                else if (isOccupied({ p.x, p.y + i }, rew)) {
                    legalMoves.emplace_back(move({ p, {p.x, p.y + i}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
        for (uint i = 1; i < 8; i++) {
            if (p.y >= i) {
                uint rew;
                if (isEmpty({ p.x, p.y - i })) {
                    legalMoves.emplace_back(move({ p, {p.x, p.y - i}, 0 }));
                }
                else if (isOccupied({ p.x, p.y - i }, rew)) {
                    legalMoves.emplace_back(move({ p, {p.x, p.y - i}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
        for (uint i = 1; i < 8; i++) {
            if (p.x + i < 8) {
                uint rew;
                if (isEmpty({ p.x + i, p.y })) {
                    legalMoves.emplace_back(move({ p, {p.x + i, p.y}, 0 }));
                }
                else if (isOccupied({ p.x + i, p.y }, rew)) {
                    legalMoves.emplace_back(move({ p, {p.x + i, p.y}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
        for (uint i = 1; i < 8; i++) {
            if (p.x >= i) {
                uint rew;
                if (isEmpty({ p.x - i, p.y })) {
                    legalMoves.emplace_back(move({ p, {p.x - i, p.y}, 0 }));
                }
                else if (isOccupied({ p.x - i, p.y }, rew)) {
                    legalMoves.push_back(move({ p, {p.x - i, p.y}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
    }
    else if (player._king.get(p.x, p.y)) {
        for (int i_x = (int)p.x - 1; i_x <= (int)p.x + 1; i_x++) {
            for (int i_y = (int)p.y - 1; i_y <= (int)p.y + 1; i_y++) {
                if (isValid(i_x, i_y)) {
                    uint rew;
                    if (isOccupied({ (uint)i_x, (uint)i_y }, rew) || isEmpty({ (uint)i_x, (uint)i_y })) {
                        legalMoves.push_back(move({ p, {(uint)i_x, (uint)i_y}, rew }));
                    }
                }
            }
        }
    }
    else if (player._bishop.get(p.x, p.y)) {
        for (uint i = 1; i < 8; i++) {
            if (isValid(p.x + i, p.y + i)) {
                uint rew;
                if (isEmpty({ p.x + i, p.y + i })) {
                    legalMoves.push_back(move({ p, {p.x + i, p.y + i}, 0 }));
                }
                else if (isOccupied({ p.x + i, p.y + i }, rew)) {
                    legalMoves.push_back(move({ p, {p.x + i, p.y + i}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
        for (uint i = 1; i < 8; i++) {
            if (isValid(p.x - i, p.y + i)) {
                uint rew;
                if (isEmpty({ p.x - i, p.y + i })) {
                    legalMoves.push_back(move({ p, {p.x - i, p.y + i}, 0 }));
                }
                else if (isOccupied({ p.x - i, p.y + i }, rew)) {
                    legalMoves.push_back(move({ p, {p.x - i, p.y + i}, rew }));
                    break;
                }
                else {
                    break;
                }
            }
        }
    }
    else if (player._knight.get(p.x, p.y)) {
        if (isValid(p.x + 1, p.y + 2)) {
            uint rew;
            if (isOccupied({ p.x + 1, p.y + 2 }, rew) || isEmpty({ p.x + 1, p.y + 2 })) {
                legalMoves.push_back(move({ p, {p.x + 1, p.y + 2}, rew }));
            }
        }
        if (isValid(p.x + 1, p.y - 2)) {
            uint rew;
            if (isOccupied({ p.x + 1, p.y - 2 }, rew) || isEmpty({ p.x + 1, p.y - 2 })) {
                legalMoves.push_back(move({ p, {p.x + 1, p.y - 2}, rew }));
            }
        }
        if (isValid(p.x - 1, p.y + 2)) {
            uint rew;
            if (isOccupied({ p.x - 1, p.y + 2 }, rew) || isEmpty({ p.x - 1, p.y + 2 })) {
                legalMoves.push_back(move({ p, {p.x - 1, p.y + 2}, rew }));
            }
        }
        if (isValid(p.x - 1, p.y - 2)) {
            uint rew;
            if (isOccupied({ p.x - 1, p.y - 2 }, rew) || isEmpty({ p.x - 1, p.y - 2 })) {
                legalMoves.push_back(move({ p, {p.x - 1, p.y - 2}, rew }));
            }
        }
        if (isValid(p.x + 2, p.y + 1)) {
            uint rew;
            if (isOccupied({ p.x + 2, p.y + 1 }, rew) || isEmpty({ p.x + 2, p.y + 1 })) {
                legalMoves.push_back(move({ p, {p.x + 2, p.y + 1}, rew }));
            }
        }
        if (isValid(p.x + 2, p.y - 1)) {
            uint rew;
            if (isOccupied({ p.x + 2, p.y - 1 }, rew) || isEmpty({ p.x + 2, p.y - 1 })) {
                legalMoves.push_back(move({ p, {p.x + 2, p.y - 1}, rew }));
            }
        }
        if (isValid(p.x - 2, p.y + 1)) {
            uint rew;
            if (isOccupied({ p.x - 2, p.y + 1 }, rew) || isEmpty({ p.x - 2, p.y + 1 })) {
                legalMoves.push_back(move({ p, {p.x - 2, p.y + 1}, rew }));
            }
        }
        if (isValid(p.x - 2, p.y - 1)) {
            uint rew;
            if (isOccupied({ p.x - 2, p.y - 1 }, rew) || isEmpty({ p.x - 2, p.y - 1 })) {
                legalMoves.push_back(move({ p, {p.x - 2, p.y - 1}, rew }));
            }
        }
    }
    return legalMoves;
}

void state::makeMove(move& m) {
    if (!isLegalMove(m)) {
        return;
    }
    player& current = players[turn];
    for (uint i = 0; i < 4; i++) {
        if (i == turn) {
            continue;
        }
        player& p = players[i];
        if (p._pawn.get(m.to.x, m.to.y)) {
            p._pawn.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 1;
            no_progress_count = 0;
        }
        else if (p._rook.get(m.to.x, m.to.y)) {
            p._rook.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 5;
            no_progress_count = 0;
        }
        else if (p._knight.get(m.to.x, m.to.y)) {
            p._knight.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 3;
            no_progress_count = 0;
        }
        else if (p._bishop.get(m.to.x, m.to.y)) {
            p._bishop.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 5;
            no_progress_count = 0;
        }
        else if (p._king.get(m.to.x, m.to.y)) {
            p._king.set(m.to.x, m.to.y, false);
            p.active = false;
            current.score += 10;
            no_progress_count = 0;
        }
    }
    if (current._pawn.get(m.from.x, m.from.y)) {
        current._pawn.set(m.from.x, m.from.y, false);
        //can be promoted to rook if y == 7
        if (m.to.y == 7) {
            current._rook.set(m.to.x, m.to.y, true);
        }
        else {
            current._pawn.set(m.to.x, m.to.y, true);
        }
    }
    else if (current._rook.get(m.from.x, m.from.y)) {
        current._rook.set(m.from.x, m.from.y, false);
        current._rook.set(m.to.x, m.to.y, true);
    }
    else if (current._knight.get(m.from.x, m.from.y)) {
        current._knight.set(m.from.x, m.from.y, false);
        current._knight.set(m.to.x, m.to.y, true);
    }
    else if (current._bishop.get(m.from.x, m.from.y)) {
        current._bishop.set(m.from.x, m.from.y, false);
        current._bishop.set(m.to.x, m.to.y, true);
    }
    else if (current._king.get(m.from.x, m.from.y)) {
        current._king.set(m.from.x, m.from.y, false);
        current._king.set(m.to.x, m.to.y, true);
    }
    nextTurn();
};

std::vector<move> state::getLegalMoves() {
    player& current = players[turn];
    Layer allPieces = current._pawn | current._rook | current._knight | current._bishop | current._king;
    std::vector<move> legalMoves;
    for (uint i = 0; i < 64; i++) {
        position p;
        p.x = i % 8;
        p.y = i / 8;
        if (allPieces.get(p.x, p.y)) {
            auto moves = getLegalMovesLocal(p);
            std::move(moves.begin(), moves.end(), std::back_inserter(legalMoves));
        }
    }
    // if more moves has same value, ensures not always first move is picked.
    std::shuffle(std::begin(legalMoves), std::end(legalMoves), global_rng());
    return legalMoves;
};

// Function to generate a mask of valid moves
py::array_t<float> state::getLegalMoveMask() {
    std::vector<float> mask(4096, 0.0f); // 64*64
    std::vector<move> legal_moves = getLegalMoves();

    // Set mask values for legal moves
    for (const auto& move : legal_moves) {
            mask[move.getIndex()] = 1.0f;
    }

    // Return mask as a NumPy array
    return py::array_t<float>(
        { 4096 },           // Shape: (4096,)
        { sizeof(float) },  // Stride: sizeof(float)
        mask.data()         // Pointer to data
        );
}

void printColor(uint c = RED) {
    if (c == RED) {
        std::cout << "R";
    }
    else if (c == YELLOW) {
        std::cout << "Y";
    }
    else if (c == GREEN) {
        std::cout << "G";
    }
    else if (c == BLUE) {
        std::cout << "B";
    }
}

void state::printBoard() {
    for (int y = 7; y >= 0; y--) {
        for (uint x = 0; x < 8; x++) {
            bool empty = true;
            for (uint i = 0; i < 4; i++) {
                if (players[i]._pawn.get(x, y)) {
                    printColor(i);
                    std::cout << "p";
                    empty = false;
                }
                if (players[i]._rook.get(x, y)) {
                    printColor(i);
                    std::cout << "r";
                    empty = false;
                }
                if (players[i]._knight.get(x, y)) {
                    printColor(i);
                    std::cout << "n";
                    empty = false;
                }
                if (players[i]._bishop.get(x, y)) {
                    printColor(i);
                    std::cout << "b";
                    empty = false;
                }
                if (players[i]._king.get(x, y)) {
                    printColor(i);
                    std::cout << "k";
                    empty = false;
                }
            }
            if (empty) {
                std::cout << "..";
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }
};

std::string getColor(uint color) {
    // Get the color name
    switch (color) {
    case RED: return "RED";
    case YELLOW: return "YELLOW";
    case GREEN: return "GREEN";
    case BLUE: return "BLUE";
    default: D_ASSERT(false, "wrong color");
    }
};

void state::printScore() const {
    // Print the score
    std::cout << "Score:\n";
    for (uint i = 0; i < 4; i++) {
        std::cout << getColor(players[i].color) << ": " << players[i].score << "\n";
    }
};

void state::printTurn() {
    // Print the turn
    std::cout << "Turn: " << getColor(turn) << "\n";
};

void state::printLegalMoves() {
    // Print the legal moves
    std::cout << "Legal moves:\n";
    auto moves = getLegalMoves();
    for (auto& move : moves) {
        std::cout << "(" << move.from.x << ", " << move.from.y << ") -> (" << move.to.x << ", " << move.to.y << "), r=" << move.reward << "\n";
    }
};

// Gives last T states of the game, skip given number of last states
py::array_t<float> states_to_numpy(const std::vector<state> &states, int T, int skip = 0) {
    // 1 plane = 64
    // 4 planes representing players score
    // 4 planes prepresenting if player is active
    // 4 planes prepresenting one-hot eoncoded turn 
    std::vector<float> flattened(1280 * T + 64 * 13, 0.0f); // 1 plane = 64, 4 planes 

    int game_size = (int) states.size() - skip - 1;

    for (int i = 0; i < T; i++) {
        if (game_size < i) break;
        auto& iterated_state = states[game_size - i];
        int layer_index = 1280 * (T - i - 1);
        auto& players = iterated_state.players;

        for (int i = 0; i < 4; i++) {
            auto& player = players[(iterated_state.turn + i) % 4];
            layer_index = fill_layer(player._pawn.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._rook.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._knight.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._bishop.getBoard(), flattened, layer_index);
            layer_index = fill_layer(player._king.getBoard(), flattened, layer_index);
        }
    }
    auto& last_state = states[game_size];

    //score planes and active planes
    for (int p = 0; p < 4; p++) {
        for (int i = 0; i < 64; i++) {
            flattened[1280 * T + p * 64 + i] = (float)last_state.players[(last_state.turn + p) % 4].score;
            flattened[1280 * T + 256 + p * 64 + i] = (float)last_state.players[(last_state.turn + p) % 4].active;
        }
    }

    //player on turn one-hot planes
    for (int i = 0; i < 64; i++) {
        flattened[1280 * T + 512 + last_state.turn * 64 + i] = 1.0f;
    }

    //moves without progress count
    for (int i = 0; i < 64; i++) {
        flattened[1280 * T + 768 + i] = (float) last_state.no_progress_count;
    }

    // Return as a NumPy array
    return py::array_t<float>(
        { 20 * T + 13, 8, 8 },     // Shape (5 channels for each player + describing last state (4 score, 4 active, 4 turn, 1 no change count), 8, 8)
        { 8 * 8 * sizeof(float),   // Stride for c
         8 * sizeof(float),        // Stride for w
         sizeof(float) },          // Stride for h
        flattened.data()           // Pointer to data
        );
}

// Struct to represent MCTS nodes
struct MCTSNode {
    std::unordered_map<move, std::array<float, 4>, move_hash> W;  // Total value of each action
    std::unordered_map<move, int, move_hash> N;    // Visit count of each action
    std::unordered_map<move, float, move_hash> P;    // probability of taking each action
    std::unordered_map<move, std::shared_ptr<MCTSNode>, move_hash> children; // Child nodes for each action
    MCTSNode* parent = nullptr; // Pointer to the parent node
    state current_state;        // Game state at this node
    bool is_terminal = false;   // Whether this is a terminal node
    bool is_leaf = true;   // Whether this is a leaf node
    std::array<float, 4> value{ 0.0f, 0.0f, 0.0f, 0.0f };         // Value of this node (only used for backpropagation), rotated to current player
    int N_total = 0;

    MCTSNode(const state& s, MCTSNode* p = nullptr) : current_state(s), parent(p) {}

    MCTSNode() = default;

    
    //PUCT (https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)
    move select_best_action(float c_puct = 4.0f) const { //4 is suggestion from medium.com article
        move best_action;
        double best_score = -std::numeric_limits<double>::infinity();

        for (auto it = N.begin(); it != N.end(); ++it) {
            auto a = it->first;  // action
            auto n = it->second; // number of visits

            double score = std::get<0>(W.at(a)) / (n + 1e-6) + (c_puct * P.at(a) * sqrt(N_total)) / (n + 1 + 1e-6);
            if (score > best_score) {
                best_score = score;
                best_action = a;
            }
        }
        return best_action;
    }

    // Expand a node by adding child nodes for all legal moves
    //processes P - multiply by mask, normalize and add dirichlet if in root
    void expand(float* P_ptr, bool add_dirichlet = false) {
        auto moves = current_state.getLegalMoves();
        float p_sum = 0.0f;
        for (auto& move : moves) {
            if (children.find(move) == children.end()) {
                children.emplace(move, std::make_shared<MCTSNode>(current_state, this));
                auto& next_node = children[move];
                next_node->current_state.step(move);
                //std::cout << "Setting: " << move << ", Value: " << children[move] << "\n";
                if (next_node->current_state.finished) {
                    next_node->is_terminal = true;
                    next_node->value = next_node->current_state.getFinalReward();
                }
                N[move] = 0;
                W[move] = { 0.0f, 0.0f, 0.0f, 0.0f };
                if (P_ptr) // can occur when calling step_given()
                    P[move] = P_ptr[move.getIndex()];
            }
            p_sum += P[move];
        }
        //can happen in early stages of training, set uniform probability
        if (p_sum == 0) {
            for (auto& [move, p] : P) {
                p = 1.0f;
                p_sum++;
            }
        }
        //normalizing, adding dirichlet
        if (add_dirichlet) {
            std::gamma_distribution<float> gamma_dist(0.03f , 1.0f); //alpha = 0.03 in AlphaZero
            std::vector<float> dirichlet_noise(moves.size());
            float dir_sum = 0.0f;
            //get dirichlet noise
            for (size_t i = 0; i < P.size(); i++) {
                dirichlet_noise[i] = gamma_dist(global_rng());
                dir_sum += dirichlet_noise[i];
            }
            //add noise
            size_t index = 0;
            for (auto& [move, p] : P) {
                p = 0.75f * p / p_sum  + (0.25f * dirichlet_noise[index++] / dir_sum);
            }
        }
        else {
            for (auto& [move, p] : P) {
                p /= p_sum;
            }
        }
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& W;
        ar& N;
        ar& P;
        ar& children;
        //ar& parent; - isn't necessary in game_storage
        ar& current_state;
        ar& is_terminal;
        ar& is_leaf;
        ar& value;
        ar& N_total;
    }
};

// Struct to represent the game with a vector of states
struct game {
    std::vector<state> states; //states from the start up to last seen state
    std::vector<move> game_trajectory; // moves between states
    std::vector<std::array<float, 4096>> probabilities; // policies used to choose a move
    std::shared_ptr<MCTSNode> root; // Root of the MCTS tree, last state from states
    std::shared_ptr<MCTSNode> current_search_position; // current position, which will be used when called get_evaluate_sample()
    bool finished = false;
    std::array<float, 4> final_reward = { 0.0f, 0.0f, 0.0f, 0.0f }; //starting from 1st player
    std::deque<move> mcts_trajectory; // trajectory from MCTS root up to position searched right now
    bool evaluation_game = false;

    int size() {
        return (int) states.size();
    }

    uint turn() {
        return root->current_state.turn;
    }

    state get(int n) {
        if (n >= 0)
            return states[n];
        else 
            return states[states.size() + n];
    }

    move get_action(int n) {
        if (n >= 0)
            return game_trajectory[n];
        else
            return game_trajectory[game_trajectory.size() + n];
    }

    void add_state(state new_state) {
        states.push_back(new_state);
    }

    game() {
        states.reserve(256);
        states.push_back(state());
        root = std::make_shared<MCTSNode>(state());
        current_search_position = root;
    }

    game(const std::string& filename) {
        auto ext = filename.substr(filename.size() - 4);

        if (ext == ".bin") {
            std::ifstream ifs(filename, std::ios::binary);
            if (ifs) boost::archive::binary_iarchive(ifs) >> *this;
        }
        else if (ext == ".txt") {
            std::ifstream ifs(filename, std::ios::binary);
            if (ifs) boost::archive::text_iarchive(ifs) >> *this;
        }
        else if (ext == ".zip") {
            int err = 0;
            zip_stat_t stat;
            zip_stat_init(&stat);
            zip_t* archive = zip_open(filename.c_str(), ZIP_RDONLY, &err);
            zip_file_t* file = zip_fopen(archive, "game.txt", 0);
            if (file) {
                zip_stat(archive, "game.txt", 0, &stat);
                std::string buffer(stat.size, '\0');
                zip_fread(file, buffer.data(), stat.size);
                std::istringstream iss(buffer);
                boost::archive::text_iarchive(iss) >> *this;
                zip_fclose(file);
            }
            else 
                throw std::runtime_error("Unsupported zip file.");
            zip_close(archive);
        }
        else throw std::runtime_error("Unsupported file format: " + filename);
    }

    void print() {
        for (int i = 0; i < states.size(); i++) {
            states[i].printBoard();
            states[i].printScore();
            if (i < game_trajectory.size())
                std::cout << game_trajectory[i] << std::endl;
            std::cout << "----" << std::endl;
        }
    }

    // Generate input for the neural network (last n states)
    py::array_t<float> get_evaluate_sample(int T, int skip) {
        try {
            std::vector<state> last_states(states); //starts with already "played" states
            last_states.reserve(states.size() + mcts_trajectory.size());
            auto node_ptr = root.get();
            for (auto it = mcts_trajectory.begin(); it != mcts_trajectory.end(); ++it) { // add MCTS states
                auto it_find = node_ptr->children.find(*it);
                if (it_find != node_ptr->children.end())
                    node_ptr = it_find->second.get();
                else
                    break;
                last_states.push_back(node_ptr->current_state);
            }
            return states_to_numpy(last_states, T, skip);
        }
        catch (const std::exception& e) {
            std::cerr << "C++ Exception: " << e.what() << std::endl;
            throw py::value_error(e.what());  // Convert C++ exception to Python exception
        }
    }

    py::array_t<float> get_legal_moves_mask() {
        return current_search_position->current_state.getLegalMoveMask();
    }

    move find_move_for_node(const std::map<move, std::shared_ptr<MCTSNode>>& children, MCTSNode* node) { //TODO: remove?
        // Iterate through the map and compare node pointers
        for (const auto& entry : children) {
            if (entry.second.get() == node) {
                return entry.first;  // Return the move that maps to this node
            }
        }

        throw std::runtime_error("Node not found in the map");
    }

    // returns new budget (number of left rollouts)
    int give_evaluated_sample(const py::array_t<float>& P, const py::array_t<float>& V, int budget, float c_puct = 4.0f) {
        if (budget <= 0) return budget;
        /*current_search_position is already leaf node, not expanded*/

        py::buffer_info P_buf = P.request();
        py::buffer_info V_buf = V.request();

        if (P_buf.size != 4096) {
            throw std::runtime_error("Policy head output must have 4096 elements");
        }
        if (V_buf.size != 4) {
            throw std::runtime_error("Value head output must have 4 elements");
        }

        float* P_ptr = static_cast<float*>(P_buf.ptr); //const float* P_ptr = P.data(0)
        float* V_ptr = static_cast<float*>(V_buf.ptr);

        auto* node = current_search_position.get();
        node->value = { V_ptr[0], V_ptr[1], V_ptr[2], V_ptr[3] };
        node->expand(P_ptr, current_search_position == root && !evaluation_game); //adds dirichlet if at root 
        node->is_leaf = false;

        int leaf_turn = node->current_state.turn;
        for (auto it = mcts_trajectory.rbegin(); it != mcts_trajectory.rend(); ++it) { // iterate through trajectory from end to the root
            node = node->parent;
            if (!node) break;
            int turn_diff = node->current_state.turn - leaf_turn;
            auto& W_ref = node->W[*it];
            // get<n> = V_ptr[4+n+turn_current-turn_leaf % 4]
            W_ref[0] += V_ptr[(4 + turn_diff) % 4];
            W_ref[1] += V_ptr[(5 + turn_diff) % 4];
            W_ref[2] += V_ptr[(6 + turn_diff) % 4];
            W_ref[3] += V_ptr[(7 + turn_diff) % 4];
            node->N[*it]++;
            node->N_total++;
        }

        /*Start again from root and do while not reached leaf node...*/
        mcts_trajectory.clear();
        budget--;
        current_search_position = root;
        while (!current_search_position->is_leaf && budget >= 0) {
            auto next_move = current_search_position->select_best_action(c_puct);
            mcts_trajectory.push_back(next_move);
            current_search_position = current_search_position->children[next_move];
            if (current_search_position->is_terminal) { //if terminal we can determine value without evaluating by nn
                auto& values = current_search_position->value;
                    
                node = current_search_position.get();
                leaf_turn = node->current_state.turn;

                for (auto it = mcts_trajectory.rbegin(); it != mcts_trajectory.rend(); ++it) { // iterate through trajectory from end to the root
                    node = node->parent;
                    if (!node) break;
                    int turn_diff = node->current_state.turn - leaf_turn;
                    auto& W_ref = node->W[*it];
                    // get<n> = V_ptr[4+n+turn_current-turn_leaf % 4]
                    W_ref[0] += values[(4 + turn_diff) % 4];
                    W_ref[1] += values[(5 + turn_diff) % 4];
                    W_ref[2] += values[(6 + turn_diff) % 4];
                    W_ref[3] += values[(7 + turn_diff) % 4];
                    node->N[*it]++;
                    node->N_total++;
                }
                mcts_trajectory.clear();
                budget--;
                current_search_position = root;
            }
        }
        /*We are here at leaf node*/
        return budget;
    }

    bool step(move& action) {
        game_trajectory.push_back(action);
        mcts_trajectory.clear();

        root = root->children[action];
        root->parent = nullptr;
        current_search_position = root;
        states.push_back(root->current_state);
        if (root->is_terminal) {
            auto& values = root->value;
            auto turn = root->current_state.turn;
            finished = true;
            final_reward = { values[(4 - turn) % 4] , values[(5 - turn) % 4] , values[(6 - turn) % 4] , values[(7 - turn) % 4] };
            return true;
        }
        return false;
    }

    bool make_step(std::array<float, 4096> policy) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double cumulative_sum = 0.0;
        double target = dist(global_rng());
        move action;

        for (auto it = root->N.begin(); it != root->N.end(); ++it) {
            cumulative_sum += policy[it->first.getIndex()];
            if (cumulative_sum >= target) {
                action = it->first;
                break;
            }
        }

        if (!evaluation_game) //not needed when only evaluating
            probabilities.push_back(std::move(policy));
        return step(action);
    }

    bool step_given(move action) {
        if (root->is_terminal) //if terminal, do nothing
            return true;
        if (!root->current_state.isLegalMove(action)) { //check if move is valid
            throw std::runtime_error("step_given(): Given move isn't legal.");
        }
        root->expand(nullptr);
        return step(action);
    }

    // Perform a deterministic step by choosing the action with the highest N
    bool step_deterministic() {
        if (root->is_terminal) //if terminal, do nothing
            return true;

        move best_a;
        int best_n = -std::numeric_limits<int>::infinity();

        for (auto it = root->N.begin(); it != root->N.end(); ++it) {
            if (it->second > best_n) {
                best_a = it->first;
                best_n = it->second;
            }
        }
        // deterministic step is usually used only during evaluation game, generating and storing deterministic policy isn't necessary.
        return step(best_a);
    }

    // For special scenario where one wants to take deterministic move but store policy
    bool step_deterministic_not_optimized() {
        if (root->is_terminal) //if terminal, do nothing
            return true;

        move best_a;
        int best_n = -std::numeric_limits<int>::infinity();

        for (auto it = root->N.begin(); it != root->N.end(); ++it) {
            if (it->second > best_n) {
                best_a = it->first;
                best_n = it->second;
            }
        }

        if (evaluation_game) //we don't need to store a policy
            return step(best_a);

        std::array<float, 4096> policy = { 0.0f };
        policy[best_a.getIndex()] = 1.0f;
        return make_step(std::move(policy));
    }

    // Perform a stochastic step based on policy (pi ~ N^(temp))
    bool step_stochastic(double temperature = 1.0) {
        if (root->is_terminal) //if terminal, do nothing
            return true;

        std::array<float, 4096> policy = { 0.0f };
        float N_pow_total = 0.0f;

        for (auto it = root->N.begin(); it != root->N.end(); ++it) {
            policy[it->first.getIndex()] = (float) std::pow(it->second, temperature);
            N_pow_total += policy[it->first.getIndex()];
        }

        for (auto it = root->N.begin(); it != root->N.end(); ++it) {
            policy[it->first.getIndex()] /= N_pow_total;
        }

        return make_step(std::move(policy));
    }

    // Perform a random step
    bool step_random() {
        return step_stochastic(0.0f); 
    }
    
    // Gives last T states of the game, skip given number of last states
    py::array_t<float> to_numpy(int T, int skip = 0) const {
        return states_to_numpy(states, T, skip);
    }

    // equivalent of: std::tuple<py::array_t<float>, std::array<float, 4096>, std::array<float, 4>>
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> get_sample(int T, int skip = 0) {
        auto index_last = states.size() - (skip + 2); //(+1) size is last_index+1, (+1) last state in sample is last state before terminated
        if(index_last < 0) throw std::out_of_range("get_sample(): skip argument out of bounds");
        auto& values = root->value;
        auto& turn = root->current_state.turn;
        auto last_turn = states[index_last].turn;
        std::array<float, 4> value = { final_reward[(4 + last_turn) % 4], final_reward[(5 + last_turn) % 4], final_reward[(6 + last_turn) % 4], final_reward[(7 + last_turn) % 4] };
        return { 
            states_to_numpy(states, T, skip + 1),  //(+1) We don't want sample where last state is terminated
            py::array_t<float>(4096, probabilities[index_last].data()),
            py::array_t<float>(4, value.data())
        };
    }

    // equivalent of: std::tuple<py::array_t<float>, std::array<float, 4096>, std::array<float, 4>>
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> get_random_sample(int T) {
        // (-1) for states.size() = x can be optained x - 1 samples
        // (-1) for last terminated state
        std::uniform_int_distribution<> dist(0, (int) states.size() - 2);
        return get_sample(T, dist(global_rng()));
    }

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& states;
        ar& game_trajectory;
        ar& probabilities;
        ar& root;
        ar& current_search_position;
        ar& finished;
        ar& final_reward;
        ar& mcts_trajectory;
        ar& evaluation_game;
    }

    void save_game(const std::string& filename) {
        std::string tmp_file = filename + ".tmp";
        auto ext = filename.substr(filename.size() - 4);

        if (ext == ".bin" || ext == ".txt") {
            std::ofstream ofs;
            ofs.open(tmp_file, std::ios::binary);
            if (!ofs) throw std::runtime_error("Failed to open file for saving: " + tmp_file);
            if (ext == ".bin") {
                boost::archive::binary_oarchive oa(ofs);
                oa << *this;
            }
            else {
                boost::archive::text_oarchive oa(ofs);
                oa << *this;
            }
            ofs.close();
            std::filesystem::rename(tmp_file, filename);
        }
        else if (ext == ".zip") {
            std::ostringstream oss;
            boost::archive::text_oarchive(oss) << *this;
            std::string data = oss.str();

            int error = 0;
            zip_t* archive = zip_open(tmp_file.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
            if (!archive) throw std::runtime_error("Failed to open file for saving: " + tmp_file);
            zip_source_t* source = zip_source_buffer(archive, data.data(), data.size(), 0);
            zip_file_add(archive, "game.txt", source, ZIP_FL_ENC_UTF_8);
            zip_close(archive);
            std::filesystem::rename(tmp_file, filename);
        }
        else throw std::runtime_error("Unsupported file format: " + filename + ". Supported: .bin, .txt, .zip");
    }

    std::string get_chess_com_representation() {
        std::string repr_str = "[StartFen4 \"R-0,0,0,0-0,0,0,0-0,0,0,0-0,0,0,0-0-{'dim':'8x8','boxOffset':1}-x,x,x,x,x,x,x,x,x,x,x,x,x,x/x,x,x,x,x,x,x,x,x,x,x,x,x,x/x,x,x,x,x,x,x,x,x,x,x,x,x,x/x,x,x,bR,bP,2,yK,yB,yN,yR,x,x,x/x,x,x,bN,bP,2,yP,yP,yP,yP,x,x,x/x,x,x,bB,bP,6,x,x,x/x,x,x,bK,bP,6,x,x,x/x,x,x,6,gP,gK,x,x,x/x,x,x,6,gP,gB,x,x,x/x,x,x,rP,rP,rP,rP,2,gP,gN,x,x,x/x,x,x,rR,rN,rB,rK,2,gP,gR,x,x,x/x,x,x,x,x,x,x,x,x,x,x,x,x,x/x,x,x,x,x,x,x,x,x,x,x,x,x,x/x,x,x,x,x,x,x,x,x,x,x,x,x,x\"][Variant \"FFA\"][RuleVariants \"CaptureTheKing DeadKingWalking EnPassant PointsForMate=10 PromoteTo=R\"][CurrentMove \"0\"][TimeControl \"1 | 7\"]\n";
        uint turn = 4;
        uint row = 0;
        for (size_t i = 0; i < game_trajectory.size(); i++) {
            auto& s = states[i];
            auto& m = game_trajectory[i];
            if (s.turn < turn) {
                repr_str += '\n' + std::to_string(++row) + ". ";
            }
            else {
                repr_str += " .. ";
            }
            turn = s.turn;
            if (turn % 2 == 0) {
                repr_str += notation_x[turn][m.from.x];
                repr_str += notation_y[turn][m.from.y];
                repr_str += '-';
                repr_str += notation_x[turn][m.to.x];
                repr_str += notation_y[turn][m.to.y];
            }
            else {
                repr_str += notation_y[turn][m.from.y];
                repr_str += notation_x[turn][m.from.x];
                repr_str += '-';
                repr_str += notation_y[turn][m.to.y];
                repr_str += notation_x[turn][m.to.x];
            }
        }
        repr_str += '\n';
        return repr_str;
    }

    void print_statistics() {
        std::cout << "Turn: " << root->current_state.turn << std::endl;
        std::cout << "N\tW\tP\tmove" << std::endl;
        for (auto& move : root->current_state.getLegalMoves()) {
            std::cout << root->N[move] << '\t' << std::get<0>(root->W[move]) << '\t' << root->P[move] << '\t' << move << std::endl;
        }
    }

};

game load_game(const std::string& filename) {
    return game(filename);
}

struct game_storage {
    std::deque<game> games;
    size_t max_size;

    game_storage(size_t max_size = 100000) : max_size(max_size) {}

    void add_game(game& g) {
        if (games.size() >= max_size) {
            games.pop_front();
        }
        games.push_back(g);
    }

    void load_game(const std::string& filename) {
        if (games.size() >= max_size) {
            games.pop_front();
        }
        games.push_back(filename); //calls constructor which loads binary file
    }

    //shouldn't be used during training, game can be deleted before calling some other game function
    game get_game(int index) {
        if (index >= 0) {
            if (index >= games.size()) throw std::out_of_range("get_game(): Index out of bounds");
            return games[index];
        }
        else {
            if (- index > games.size()) throw std::out_of_range("get_game(): Index out of bounds");
            return games[games.size() + index];
        } 
    }

    //get random sample
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> get_random_sample(int T = 8, bool winrate_sampling = true, float win = 0.5f) {
        std::vector<int> game_sizes(games.size());
        for (size_t i = 0; i < games.size(); i++) {
            game_sizes[i] = games[i].size() - 2; //number of different samples we can obtain from this game
        }
        std::discrete_distribution<> dist(game_sizes.begin(), game_sizes.end());
        if(!winrate_sampling)
            return games[dist(global_rng())].get_random_sample(T);
        else {
            std::bernoulli_distribution dist_value(win);
            bool value = dist_value(global_rng());
            while (true) {
                auto sample = games[dist(global_rng())].get_random_sample(T);

                auto& value_output = std::get<2>(sample);
                py::buffer_info value_output_buf = value_output.request();
                float* value_output_ptr = static_cast<float*>(value_output_buf.ptr);

                if ((value && value_output_ptr[0] >= 0.0f) || (!value && value_output_ptr[0] <= 0.0f)) // player is winning/loosing
                    return sample;
            }
        }
    }

    //get random sample given probability distribution of players on move
    std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> get_random_sample_distribution(int T = 8, float red = 0.25f, float blue = 0.25f, float yellow = 0.25f, float green = 0.25f, bool winrate_sampling = true, float win = 0.5f) {
        std::vector<int> game_sizes(games.size());
        for (size_t i = 0; i < games.size(); i++) {
            game_sizes[i] = games[i].size() - 2;
        }

        std::uniform_real_distribution<double> dist_target(0.0, 1.0);
        std::discrete_distribution<> dist_game(game_sizes.begin(), game_sizes.end());
        std::bernoulli_distribution dist_value(win);

        //normalise, get upper bounds
        float sum = red + blue + yellow + green;
        float bounds[4];
        bounds[0] = red / sum;
        bounds[1] = bounds[0] + (blue / sum);
        bounds[2] = bounds[1] + (yellow / sum);
        bounds[3] = bounds[2] + (green / sum);

        double target = dist_target(global_rng());
        uint turn = 0;
        for (uint i = 4; i > 0; i--) {
            if (target <= bounds[i-1]) {
                turn = i-1;
            }
        }

        bool value = dist_value(global_rng()); // true means returning "winning" position, false returning "loosing" position ("winning" -> value >= 0) 
        
        while (true) { //sample until sampled one with last turn with specific color.
            auto sample = games[dist_game(global_rng())].get_random_sample(T);
            
            auto& input = std::get<0>(sample);
            py::buffer_info input_buf = input.request();
            float* input_ptr = static_cast<float*>(input_buf.ptr);

            auto& value_output = std::get<2>(sample);
            py::buffer_info value_output_buf = value_output.request();
            float* value_output_ptr = static_cast<float*>(value_output_buf.ptr);

            if (input_ptr[1280 * T + 512 + turn * 64] == 1.0f) { // player on turn
                if(!winrate_sampling || (value && value_output_ptr[0] >= 0.0f) || (!value && value_output_ptr[0] <= 0.0f)) // player is winning/loosing
                    return sample;
            }
        }
    }

    size_t size() { 
        return games.size(); 
    }
};

PYBIND11_MODULE(chaturajienv, m) {
    m.doc() = "Chaturaji environment";
    py::class_<Layer>(m, "Layer")
        .def(pybind11::init<>())
        .def(pybind11::init<uint64_t>())
        .def("set", &Layer::set)
        .def("get", &Layer::get)
        .def("rotate_to_general", &Layer::rotate_to_general)
        .def("rotate_to_local", &Layer::rotate_to_local)
        .def("getBoard", &Layer::getBoard)
        .def("countBits", &Layer::countBits)
        .def("print", &Layer::print);
    py::class_<player>(m, "player")
        .def(pybind11::init<uint>())
        .def_readwrite("_pawn", &player::_pawn)
        .def_readwrite("_rook", &player::_rook)
        .def_readwrite("_knight", &player::_knight)
        .def_readwrite("_bishop", &player::_bishop)
        .def_readwrite("_king", &player::_king)
        .def_readwrite("_attacking", &player::_attacking)
        .def_readwrite("score", &player::score)
        .def_readwrite("color", &player::color)
        .def_readwrite("active", &player::active);
    py::class_<state>(m, "state")
        .def(pybind11::init<>())
        .def_readwrite("players", &state::players)
        .def_readwrite("turn", &state::turn)
        .def_readwrite("finished", &state::finished)
        .def("nextTurn", &state::nextTurn)
        .def("isEmpty", &state::isEmpty)
        .def("isOccupied", &state::isOccupied)
        .def("getLegalMovesLocal", &state::getLegalMovesLocal)
        .def("isLegalMove", &state::isLegalMove)
        .def("makeMove", &state::makeMove)
        .def("getLegalMoves", &state::getLegalMoves)
        .def("getLegalMoveMask", &state::getLegalMoveMask)
        .def("to_numpy", &state::to_numpy)
        .def("printBoard", &state::printBoard, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("printScore", &state::printScore, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("printTurn", &state::printTurn, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("printLegalMoves", &state::printLegalMoves, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("reset", &state::reset)
        .def("sample", &state::sample)
        .def("step", &state::step)
        .def("get_score", &state::get_score)
        .def("get_score_default", &state::get_score_default);
    py::class_<game>(m, "game")
        .def(pybind11::init<>())
        .def_readwrite("evaluation_game", &game::evaluation_game)
        .def("size", &game::size)
        .def("get", &game::get)
        .def("turn", &game::turn)
        .def("get_action", &game::get_action)
        .def_readonly("final_reward", &game::final_reward)
        .def("add_state", &game::add_state)
        .def("get_evaluate_sample", &game::get_evaluate_sample, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("give_evaluated_sample", &game::give_evaluated_sample, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_legal_moves_mask", &game::get_legal_moves_mask)
        .def("step_deterministic", &game::step_deterministic, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("step_deterministic_not_optimized", &game::step_deterministic_not_optimized, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("step_stochastic", &game::step_stochastic, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("step_random", &game::step_random, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("step_given", &game::step_given, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("to_numpy", &game::to_numpy)
        .def("get_sample", &game::get_sample, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_random_sample", &game::get_random_sample)
        .def("save_game", &game::save_game)
        .def("get_chess_com_representation", &game::get_chess_com_representation)
        .def("print", &game::print, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("print_statistics", &game::print_statistics, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    py::class_<game_storage>(m, "game_storage")
        .def(py::init<size_t>(), py::arg("max_size") = 100000)
        .def("add_game", &game_storage::add_game)
        .def("load_game", &game_storage::load_game)
        .def("get_game", &game_storage::get_game)
        .def("get_random_sample", &game_storage::get_random_sample, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("get_random_sample_distribution", &game_storage::get_random_sample_distribution, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("size", &game_storage::size);  //number of stored games
    py::class_<position>(m, "position")
        .def(pybind11::init<uint, uint>())
        .def_readwrite("x", &position::x)
        .def_readwrite("y", &position::y)
        .def("__repr__", [](const position& p) {
            return "(" + std::to_string(p.x) + "," + std::to_string(p.y) + ")";
        });
    py::class_<move>(m, "move")
        .def(pybind11::init<position, position, uint>())
        .def_readwrite("fr", &move::from) //from isn't accepted in python
        .def_readwrite("to", &move::to)
        .def_readwrite("reward", &move::reward)
        .def("__repr__", [](const move& m) {
            return "<(" + std::to_string(m.from.x) + "," + std::to_string(m.from.y) +
            ")->(" + std::to_string(m.to.x) + "," + std::to_string(m.to.y) +
            ")," + std::to_string(m.reward) + ">";
         });
    m.def("printColor", &printColor, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());
    m.def("getColor", &getColor);
    m.def("make", &make);
    m.def("load_game", &load_game);
}

int main(int argc, char* argv[]) {
    // load bin game and output chess.com representation
    if (argc == 3 && (std::string(argv[1]) == "-g" || std::string(argv[1]) == "--game")) {
        std::string filename = argv[2];
        game g = load_game(filename);
        std::cout << g.get_chess_com_representation() << std::endl;
        return 0;
    }

    state s;
    s.printScore();
    s.printTurn();
    s.printBoard();
    s.printLegalMoves();
    for (int i = 0; i < 100; i++) {
        std::cout << "Enter move (from_x from_y to_x to_y): ";
        move m;
        std::cin >> m.from.x >> m.from.y >> m.to.x >> m.to.y;
        s.makeMove(m);
        s.printScore();
        s.printTurn();
        s.printBoard();
        s.printLegalMoves();
        if (s.finished) {
            std::cout << "Game over\n";
            for (auto& player : s.players) {
                if (player.active) {
                    std::cout << getColor(player.color) << " wins\n";
                }
            }
            for (auto& player : s.players) {
                std::cout << getColor(player.color) << " score: " << player.score << "\n";
            }
            break;
        }
    }
    return 0;
}
