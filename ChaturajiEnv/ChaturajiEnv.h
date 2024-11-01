#pragma once

#ifndef CHATURAJI_H
#define CHATURAJI_H

#include <iostream>
#include <python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <vector>
#include <set>
#include <map>
#include <memory>

// Define uint_t alias
using uint = unsigned short;

void print_color(uint color, std::ostream& s = std::cout);

// Struct declarations
struct position {
    uint x;
    uint y;

    position(uint _x = 0, uint _y = 0);

    bool is_valid() const;
    position rotate_to_chessboard(const uint& color) const;
    bool operator<(const position& other) const;
    bool operator==(const position& other) const;
    friend std::ostream& operator<<(std::ostream& os, const position& pos);
};

position rotate_to_chessboard(position pos, const uint& color);
position rotate_to_local(position pos, const uint& color);

struct player {
    uint score;
    uint color;
    bool active;
    bool out_of_time;

    player(uint c = 0, uint s = 0, bool active = true, bool out_of_time = false);
};

struct action {
    position from;
    position to;

    action(position from_p = { 0, 0 }, position to_p = { 0, 0 });
    bool is_pass();
    friend std::ostream& operator<<(std::ostream& os, const action& a);
};

struct chessboard;

struct piece {
    uint color;
    position pos; // local position

    piece(uint c = 4, position p = {0,0});

    uint get_color();
    virtual uint get_value();
    virtual std::set<position> get_moves(chessboard& board);
    virtual void print_piece(std::ostream& s);
    void print(std::ostream& s);
};

using board_map = std::map<position, std::unique_ptr<piece>>;

struct chessboard {
    board_map board;
    player players[4];
    uint turn;

    chessboard();
    void next_turn();
    bool is_current_player_active();
    player& current_player();
    uint count_active();
    bool make_move(action a);

    std::map<position, piece*> get_board() const;
};

class ChaturajiEnv {
public:
    ChaturajiEnv();
    void reset();
    std::tuple<std::set<position>, std::set<position>, std::set<position>, std::set<position>,
        std::set<position>, std::set<position>, std::set<position>, std::set<position>>
        get_possible_actions();
    std::set<position> get_possible_actions_from(position from);
    std::tuple<bool, uint, bool, std::string> step(action a);
    void render(std::ostream& s = std::cout);
    void render_p() {
        render();
    }
    void close(std::ostream& s = std::cout);
    const chessboard& get_board_reference() const {
        return _board;
    }
    std::map<position, piece*> get_board() const {
        return _board.get_board();
    }
private:
    chessboard _board;
    uint episode_length = 0;
    std::vector<action> trajectory;
};

// Piece-derived classes
struct pawn : piece {
    uint index;

    pawn(uint c, position p, uint i);
    uint get_value() override;
    std::set<position> get_moves(chessboard& board) override;
    void print_piece(std::ostream& s) override;
};

struct knight : piece {
    using piece::piece;

    uint get_value() override;
    std::set<position> get_moves(chessboard& board) override;
    void print_piece(std::ostream& s) override;
};

struct bishop : piece {
    using piece::piece;

    uint get_value() override;
    std::set<position> get_moves(chessboard& board) override;
    void print_piece(std::ostream& s) override;
};

struct rock : piece {
    using piece::piece;

    uint get_value() override;
    std::set<position> get_moves(chessboard& board) override;
    void print_piece(std::ostream& s) override;
};

struct king : piece {
    using piece::piece;

    uint get_value() override;
    std::set<position> get_moves(chessboard& board) override;
    void print_piece(std::ostream& s) override;
};

#endif // CHATURAJI_H

