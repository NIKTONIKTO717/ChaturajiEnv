#include <iostream>
#include <python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <map>

namespace py = pybind11;
using uint8 = uint8_t;

#define u(x) (static_cast<uint8_t>(x))

struct position {
    uint8 x;
    uint8 y;

    bool operator<(const position& other) const {
        return 8*y+x < 8*other.y + other.x;
    }
};

position rotate_to_local(position pos, uint8 &color) {
    if (color == 0) {
        return pos;
    }
    else if (color == 1) {
        return {(uint8)(7-pos.y), (uint8)(pos.x)};
    }
    else if (color == 2) {
        return { (uint8)(7 - pos.x), (uint8)(7 - pos.y) };
    }
    else
        return { (uint8)(pos.y), (uint8)(7 - pos.x) };
}

position rotate_to_chessboard(position pos, uint8& color) {
    if (color == 0) {
        return pos;
    }
    else if (color == 1) {
        return { (uint8)(pos.y), (uint8)(7 - pos.x) };
    }
    else if (color == 2) {
        return { (uint8)(7 - pos.x), (uint8)(7 - pos.y) };
    }
    else
        return { (uint8)(7 - pos.y), (uint8)(pos.x) };
}

bool is_position_valid(position pos) {
    return pos.x < 8 && pos.y < 8;
}

struct player {
    unsigned score = 0;
    uint8 color = 0;
    bool active = true;
    bool out_of_time = false;
};

struct piece;

struct chessboard {
    std::map<position, piece> board;
    player players[4];
    uint8 turn = 0;

    void next_turn() {
        turn = (turn + 1) & 0x03;
    }
};

struct piece {
    uint8 color = 0;
    position pos = {0,0};

    uint8 get_color() {
        return color;
    }

    virtual uint8 get_value() {
        return 0;
    }

    virtual std::vector<position> get_moves(chessboard& board) {
        return std::vector<position>();
    }
};

struct pawn : piece {
    uint8 get_value() override {
        return 1;
    }
    std::vector<position> get_moves(chessboard& board) override {
        position moves[3] = { {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x), u(pos.y + 1)} , {u(pos.x + 1), u(pos.y + 1)} };
        std::vector<position> valid_moves;
        if (is_position_valid(rotate_to_chessboard(moves[0], color)) && board.board.count(moves[0]))
            valid_moves.push_back(std::move(moves[0]));
        if (is_position_valid(rotate_to_chessboard(moves[1], color)) && !board.board.count(moves[1]))
            valid_moves.push_back(std::move(moves[1]));
        if (is_position_valid(rotate_to_chessboard(moves[2], color)) && board.board.count(moves[2]))
            valid_moves.push_back(std::move(moves[2]));
        return valid_moves;
    }
};

struct knight : piece {
    uint8 get_value() override {
        return 3;
    }
    std::vector<position> get_moves(chessboard& board) override {
        std::vector<position> valid_moves;
        position moves[8] = {   {u(pos.x + 2), u(pos.y + 1)} , {u(pos.x + 2), u(pos.y - 1)} ,
                                {u(pos.x - 2), u(pos.y + 1)} , {u(pos.x - 2), u(pos.y - 1)} ,
                                {u(pos.x + 1), u(pos.y + 2)} , {u(pos.x + 1), u(pos.y - 2)} ,
                                {u(pos.x - 1), u(pos.y + 2)} , {u(pos.x - 1), u(pos.y - 2)} };
        for (auto &move : moves) {
            if (is_position_valid(rotate_to_chessboard(move, color))) valid_moves.push_back(std::move(move));
        }
        return valid_moves;
    }
};

struct bishop : piece {
    uint8 get_value() override {
        return 3;
    }
    std::vector<position> get_moves(chessboard& board) override {
        std::vector<position> valid_moves;
        position moves_up_left[7] = { {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x - 2), u(pos.y + 2)} ,
                                      {u(pos.x - 3), u(pos.y + 3)} , {u(pos.x - 4), u(pos.y + 4)} ,
                                      {u(pos.x - 5), u(pos.y + 5)} , {u(pos.x - 6), u(pos.y + 6)} ,
                                      {u(pos.x - 7), u(pos.y + 7)} };
        position moves_up_right[7] = { {u(pos.x + 1), u(pos.y + 1)} , {u(pos.x + 2), u(pos.y + 2)} ,
                                       {u(pos.x + 3), u(pos.y + 3)} , {u(pos.x + 4), u(pos.y + 4)} ,
                                       {u(pos.x + 5), u(pos.y + 5)} , {u(pos.x + 6), u(pos.y + 6)} ,
                                       {u(pos.x + 7), u(pos.y + 7)} };
        position moves_down_left[7] = { {u(pos.x - 1), u(pos.y - 1)} , {u(pos.x - 2), u(pos.y - 2)} ,
                                        {u(pos.x - 3), u(pos.y - 3)} , {u(pos.x - 4), u(pos.y - 4)} ,
                                        {u(pos.x - 5), u(pos.y - 5)} , {u(pos.x - 6), u(pos.y - 6)} ,
                                        {u(pos.x - 7), u(pos.y - 7)} };
        position moves_down_right[7] = { {u(pos.x + 1), u(pos.y - 1)} , {u(pos.x + 2), u(pos.y - 2)} ,
                                         {u(pos.x + 3), u(pos.y - 3)} , {u(pos.x + 4), u(pos.y - 4)} ,
                                         {u(pos.x + 5), u(pos.y - 5)} , {u(pos.x + 6), u(pos.y - 6)} ,
                                         {u(pos.x + 7), u(pos.y - 7)} };
        for (auto& move : moves_up_left) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto& move : moves_up_right) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto& move : moves_down_left) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto& move : moves_down_right) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        return valid_moves;
    }
};

struct rock : piece {
    uint8 get_value() override {
        return 5;
    }
    std::vector<position> get_moves(chessboard& board) override {
        std::vector<position> valid_moves;
        position moves_up[7] = {    {u(pos.x), u(pos.y + 1)} , {u(pos.x), u(pos.y + 2)} ,
                                    {u(pos.x), u(pos.y + 3)} , {u(pos.x), u(pos.y + 4)} ,
                                    {u(pos.x), u(pos.y + 5)} , {u(pos.x), u(pos.y + 6)} ,
                                    {u(pos.x), u(pos.y + 7)} };
        position moves_down[7] = {  {u(pos.x), u(pos.y - 1)} , {u(pos.x), u(pos.y - 2)} ,
                                    {u(pos.x), u(pos.y - 3)} , {u(pos.x), u(pos.y - 4)} ,
                                    {u(pos.x), u(pos.y - 5)} , {u(pos.x), u(pos.y - 6)} ,
                                    {u(pos.x), u(pos.y - 7)} };
        position moves_left[7] = {  {u(pos.x - 1), u(pos.y)} , {u(pos.x - 2), u(pos.y)} ,
                                    {u(pos.x - 3), u(pos.y)} , {u(pos.x - 4), u(pos.y)} ,
                                    {u(pos.x - 5), u(pos.y)} , {u(pos.x - 6), u(pos.y)} ,
                                    {u(pos.x - 7), u(pos.y)} };
        position moves_right[7] = { {u(pos.x + 1), u(pos.y)} , {u(pos.x + 2), u(pos.y)} ,
                                    {u(pos.x + 3), u(pos.y)} , {u(pos.x + 4), u(pos.y)} ,
                                    {u(pos.x + 5), u(pos.y)} , {u(pos.x + 6), u(pos.y)} ,
                                    {u(pos.x + 7), u(pos.y)} };
        for (auto &move : moves_up) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto &move : moves_down) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto &move : moves_left) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        for (auto &move : moves_right) {
            if (is_position_valid(rotate_to_chessboard(move, color))) {
                valid_moves.push_back(std::move(move));
                if (board.board.count(move)) break;
            }
            else break;
        }
        return valid_moves;
    }
};

struct king : piece {
    uint8 get_value() override {
        return 3;
    }
    std::vector<position> get_moves(chessboard& board) override {
        std::vector<position> valid_moves;
        position moves[8] = { {u(pos.x + 1), u(pos.y + 1)} , {u(pos.x + 1), u(pos.y - 1)} ,
                                 {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x - 1), u(pos.y - 1)} ,
                                 {u(pos.x - 1), u(pos.y)} , {u(pos.x), u(pos.y - 1)} ,
                                 {u(pos.x + 1), u(pos.y)} , {u(pos.x), u(pos.y + 1)} };
        for (auto& move : moves) {
            if (is_position_valid(rotate_to_chessboard(move, color))) valid_moves.push_back(std::move(move));
        }
        return valid_moves;
    }
};

class ChaturajiEnv {
public:
    ChaturajiEnv() : position(0), goal(10), done(false) {}

    py::array_t<int> reset() {
        position = 0;
        done = false;
        return py::array_t<int>({ position });
    }

    py::tuple step(int action) {
        if (done) {
            throw std::runtime_error("Episode is done. Please reset the environment.");
        }

        // Example logic: move left (-1) or right (+1)
        if (action == 1) position++;
        else if (action == -1) position--;

        // Check if we have reached the goal
        if (position >= goal) {
            done = true;
            return py::make_tuple(py::array_t<int>({ position }), 1.0, done, py::dict());
        }

        // Provide a reward of 0 for intermediate steps
        return py::make_tuple(py::array_t<int>({ position }), 0.0, done, py::dict());
    }

    void render() const {
        std::cout << "Current position: " << position << std::endl;
    }

    void close() {
        std::cout << "Closing environment." << std::endl;
    }

private:
    int position;
    const int goal;
    bool done;
};

PYBIND11_MODULE(my_cpp_env, m) {
    py::class_<ChaturajiEnv>(m, "SimpleEnv")
        .def(py::init<>())
        .def("reset", &ChaturajiEnv::reset)
        .def("step", &ChaturajiEnv::step)
        .def("render", &ChaturajiEnv::render)
        .def("close", &ChaturajiEnv::close);
}

int main(){
    std::cout << "Hello World!\n";
}
