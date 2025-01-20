#include <cstdint>
#include <iostream>
#include <bitset>
#include <cassert>
#include <vector>
#include <tuple>
#include <array>
#include <python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

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

// Named constants for colors
constexpr uint RED = 0;
constexpr uint YELLOW = 2;
constexpr uint GREEN = 3;
constexpr uint BLUE = 1;

struct position {
    uint x;
    uint y;
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
int fill_layer(uint64_t board, std::vector<float>& flattened, int layer_index) {
    for (int i = 0; i < 64; ++i) {
        flattened[layer_index++] = (board & (1ULL << i)) ? 1.0f : 0.0f;
    }
    return layer_index;
}

struct move {
    position from;
    position to;
    uint reward = 0;

    //define operator==
    bool operator==(const move& m) const {
        return std::tie(from.x, from.y, to.x, to.y) == std::tie(m.from.x, m.from.y, m.to.x, m.to.y);
    }

    //default constructor
    move() = default;

    //define constructor
    move(position from, position to, uint reward = 0) : from(from), to(to), reward(reward) {}

    //returns index on flatten action_mask of size 64*64
    uint64_t getIndex() const {
        return ((from.y << 3) | from.x) << 6 | ((to.y << 3) | to.x);
    }
};

class Layer {
    uint64_t board; // 8x8 chessboard represented as 64 bits

public:
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
        return __builtin_popcountll(board); // GCC/Clang
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
};

//players in array, starting from red --> not starting from player on turn! 
// player boards are represented in position of next turn player, f.e. if turn = BLUE, blue is in terms of (x , y) position as red at the beginning
//turn - index of player on turn
struct state {
    std::array<player, 4> players;
    uint turn;
    bool finished = false;

    //define default constructor
    state() {
        for (uint i = 0; i < 4; i++) {
            players[i] = player(i);
        }
        turn = RED;
    }

    void nextTurn() {
        for (uint i = 0; i < 3; i++) { 
            turn = (turn + 1) % 4;
            for (auto& player : players) {
                player._pawn = player._pawn.rotate_to_local(1);
                player._rook = player._rook.rotate_to_local(1);
                player._knight = player._knight.rotate_to_local(1);
                player._bishop = player._bishop.rotate_to_local(1);
                player._king = player._king.rotate_to_local(1);
                player._attacking = player._attacking.rotate_to_local(1);
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

    void printScore();

    void printTurn();

    void printLegalMoves();

    //TODO: maybe remove?
    std::vector<bool> getState();

    //gymnasium like interface
    std::pair<std::vector<bool>, std::string> reset() {
        for (uint i = 0; i < 4; i++) {
            players[i] = player(i);
        }
        turn = RED;
        finished = false;
        std::string info = "Reset info";
        return { getState(), info };
    }

    move sample() {
        auto moves = getLegalMoves();
        return moves[rand() % moves.size()];
    }

    //observation (new state), reward, terminated, truncated, info
    std::tuple<std::vector<bool>, uint, bool, bool, std::string> step(move& m) {
        makeMove(m);
        std::string info = "Step info";
        return { getState(), m.reward, finished, false, info };
    }

    std::tuple<uint, uint, uint, uint> getScore() const {
        return { players[turn % 4].score, players[(turn + 1) % 4].score, players[(turn + 2) % 4].score, players[(turn + 3) % 4].score };
    }

    //returns score indexed from 1st player (red)
    std::tuple<uint, uint, uint, uint> getScoreDefault() const {
        return { players[0].score, players[1].score, players[2].score, players[3].score };
    }

    // Convert the state into a flat vector of floats for NumPy
    py::array_t<float> to_numpy() const {
        std::vector<float> flattened(1280, 0.0f); // Total size: 1280 = 4 * 5 * 8 * 8

        int layer_index = 0;
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
            rew = player.active ? 3 : 0;
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
        }
        else if (p._rook.get(m.to.x, m.to.y)) {
            p._rook.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 5;
        }
        else if (p._knight.get(m.to.x, m.to.y)) {
            p._knight.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 3;
        }
        else if (p._bishop.get(m.to.x, m.to.y)) {
            p._bishop.set(m.to.x, m.to.y, false);
            if (p.active) current.score += 3;
        }
        else if (p._king.get(m.to.x, m.to.y)) {
            p._king.set(m.to.x, m.to.y, false);
            p.active = false;
            current.score += 10;
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

void state::printScore() {
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

std::vector<bool> state::getState() {
    std::vector<bool> out_state(1284); //20 * 64 + 4
    for (size_t i = 0; i < 64; i++) {
        out_state[i] = players[(turn + 0) % 4]._pawn.getBoard() & (1ULL << i);
        out_state[i + 64] = players[(turn + 0) % 4]._rook.getBoard() & (1ULL << i);
        out_state[i + 128] = players[(turn + 0) % 4]._knight.getBoard() & (1ULL << i);
        out_state[i + 192] = players[(turn + 0) % 4]._bishop.getBoard() & (1ULL << i);
        out_state[i + 256] = players[(turn + 0) % 4]._king.getBoard() & (1ULL << i);
        out_state[i + 320] = players[(turn + 1) % 4]._pawn.getBoard() & (1ULL << i);
        out_state[i + 384] = players[(turn + 1) % 4]._rook.getBoard() & (1ULL << i);
        out_state[i + 448] = players[(turn + 1) % 4]._knight.getBoard() & (1ULL << i);
        out_state[i + 512] = players[(turn + 1) % 4]._bishop.getBoard() & (1ULL << i);
        out_state[i + 576] = players[(turn + 1) % 4]._king.getBoard() & (1ULL << i);
        out_state[i + 640] = players[(turn + 2) % 4]._pawn.getBoard() & (1ULL << i);
        out_state[i + 704] = players[(turn + 2) % 4]._rook.getBoard() & (1ULL << i);
        out_state[i + 768] = players[(turn + 2) % 4]._knight.getBoard() & (1ULL << i);
        out_state[i + 832] = players[(turn + 2) % 4]._bishop.getBoard() & (1ULL << i);
        out_state[i + 896] = players[(turn + 2) % 4]._king.getBoard() & (1ULL << i);
        out_state[i + 960] = players[(turn + 3) % 4]._pawn.getBoard() & (1ULL << i);
        out_state[i + 1024] = players[(turn + 3) % 4]._rook.getBoard() & (1ULL << i);
        out_state[i + 1088] = players[(turn + 3) % 4]._knight.getBoard() & (1ULL << i);
        out_state[i + 1152] = players[(turn + 3) % 4]._bishop.getBoard() & (1ULL << i);
        out_state[i + 1216] = players[(turn + 3) % 4]._king.getBoard() & (1ULL << i);
    }
    //4 bits representing if player is active
    out_state[1280] = players[(turn + 0) % 4].active;
    out_state[1281] = players[(turn + 1) % 4].active;
    out_state[1282] = players[(turn + 2) % 4].active;
    out_state[1283] = players[(turn + 3) % 4].active;
    return out_state;
}

// Struct to represent the game with a vector of states
struct game {
    std::vector<state> states;

    int getSize() {
        return states.size();
    }

    void addState(state& new_state) {
        states.push_back(new_state);
    }
    
    // Gives last T states of the game, skip given numebr of last states
    py::array_t<float> to_numpy(int T, int skip = 0) const {
        // 1 plane = 64
        // 4 planes representing players score
        // 4 planes prepresenting if player is active
        // 4 planes prepresenting one-hot eoncoded turn 
        std::vector<float> flattened(1280*T + 64*12, 0.0f); // 1 plane = 64, 4 planes 

        int game_size = states.size() - skip;

        for (int i = 0; i < T; i++) {
            if (game_size > i) break;
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
        auto& last_state = states[game_size - 1];

        //score planes and active planes
        for (int p = 0; p < 4; p++) {
            for (int i = 0; i < 64; i++) {
                flattened[1280 * T + p * 64 + i] = (float) last_state.players[(last_state.turn + p) % 4].score;
                flattened[1280 * T + 256 + p * 64 + i] = (float) last_state.players[(last_state.turn + p) % 4].active;
            }
        }

        //active player plane
        for (int i = 0; i < 64; i++) {
            flattened[1280 * T + 512 + last_state.turn*64 + i] = 1.0f;
        }

        // Return as a NumPy array
        return py::array_t<float>(
            { 20 * T + 12, 8, 8 },     // Shape (5 chanels for each player + describing last state, 8, 8)
            { 8 * 8 * sizeof(float),   // Stride for c
             8 * sizeof(float),        // Stride for w
             sizeof(float) },          // Stride for h
            flattened.data()           // Pointer to data
            );
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
        .def("getState", &state::getState)
        .def("reset", &state::reset)
        .def("sample", &state::sample)
        .def("step", &state::step)
        .def("getScore", &state::getScore)
        .def("getScoreDefault", &state::getScoreDefault);
    py::class_<game>(m, "game")
        .def(pybind11::init<>())
        .def("getSize", &game::getSize)
        .def("to_numpy", &game::to_numpy);
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
}

int main() {
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
