#include "ChaturajiEnv.h"
#include <memory>

namespace py = pybind11;
using uint = unsigned short;

#define u(x) (static_cast<unsigned short>(x))

void print_color(uint color, std::ostream& s) {
    switch (color) {
    case 0:
        s << 'r';
        break;
    case 1:
        s << 'b';
        break;
    case 2:
        s << 'y';
        break;
    case 3:
        s << 'g';
        break;
    default:
        s << '.';
    }
}

position::position(uint _x, uint _y) : x(_x), y(_y) {}

bool position::is_valid() const {
    return x < 8 && y < 8;
}

position position::rotate_to_chessboard( const uint& color) const {
    if (color == 0) {
        return { x,y }; //TODO can be replace with *this?
    }
    else if (color == 1) {
        return { (uint)(y), (uint)(7 - x) };
    }
    else if (color == 2) {
        return { (uint)(7 - x), (uint)(7 - y) };
    }
    else
        return { (uint)(7 - y), (uint)(x) };
}

bool position::operator<(const position& other) const {
    return 8*y+x < 8*other.y + other.x;
}

bool position::operator==(const position& other) const {
    return x == other.x && y == other.y;
}

std::ostream& operator<<(std::ostream& os, const position& pos) {
    os << '(' << pos.x << ',' << pos.y << ')';
    py::print(pos.x, pos.y);
    return os;
}

position rotate_to_local(position pos, const uint &color) {
    if (color == 0) {
        return pos;
    }
    else if (color == 1) {
        return {(uint)(7-pos.y), (uint)(pos.x)};
    }
    else if (color == 2) {
        return { (uint)(7 - pos.x), (uint)(7 - pos.y) };
    }
    else
        return { (uint)(pos.y), (uint)(7 - pos.x) };
}

std::ostream& operator<<(std::ostream& os, const action& a) {
    os << a.from << " -> " << a.to;
    py::print(a.from.x, a.from.y, a.to.x, a.to.y);
    return os;
}

position rotate_to_chessboard(position pos, const uint& color) {
    return pos.rotate_to_chessboard(color);
}

player::player(uint _color, uint _score, bool _active, bool _out_of_time) : color(_color), score(_score), active(_active), out_of_time(_out_of_time) {};

    action::action(position from_p, position to_p) : from(from_p), to(to_p) {}

    bool action::is_pass() {
        return from == to;
    }

    piece::piece(uint c, position p) : color(c), pos(p) {}

    uint piece::get_color() {
        return color;
    }

    uint piece::get_value() {
        return uint();
    }

    std::set<position> piece::get_moves(chessboard& board) {
        return std::set<position>();
    }

    void piece::print_piece(std::ostream& s) {
        s << '.';
    }

    void piece::print(std::ostream& s) {
        print_color(color, s);
        print_piece(s);
    }

    pawn::pawn(uint c, position p, uint i) : piece(c,p), index(i) {}

    uint pawn::get_value() {
        return 1;
    }

    std::set<position> pawn::get_moves(chessboard& board) {
        position moves[3] = { {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x), u(pos.y + 1)} , {u(pos.x + 1), u(pos.y + 1)} };
        std::set<position> valid_moves;
        if (moves[0].is_valid() && board.board[moves[0].rotate_to_chessboard(color)]->color < 4 && board.board[moves[0].rotate_to_chessboard(color)]->color != color)
            valid_moves.emplace(moves[0].rotate_to_chessboard(color));
        if (moves[1].is_valid() && board.board[moves[1].rotate_to_chessboard(color)]->color == 4)
            valid_moves.emplace(moves[1].rotate_to_chessboard(color));
        if (moves[2].is_valid() && board.board[moves[2].rotate_to_chessboard(color)]->color < 4 && board.board[moves[2].rotate_to_chessboard(color)]->color != color)
            valid_moves.emplace(moves[2].rotate_to_chessboard(color));
        return valid_moves;
    }

    void pawn::print_piece(std::ostream& s) {
        s << 'p';
    }

    uint knight::get_value() {
        return 3;
    }
    std::set<position> knight::get_moves(chessboard& board) {
        std::set<position> valid_moves;
        position moves[8] = { {u(pos.x + 2), u(pos.y + 1)} , {u(pos.x + 2), u(pos.y - 1)} ,
                                {u(pos.x - 2), u(pos.y + 1)} , {u(pos.x - 2), u(pos.y - 1)} ,
                                {u(pos.x + 1), u(pos.y + 2)} , {u(pos.x + 1), u(pos.y - 2)} ,
                                {u(pos.x - 1), u(pos.y + 2)} , {u(pos.x - 1), u(pos.y - 2)} };
        for (auto& move : moves) {
            if (move.is_valid() && board.board[move.rotate_to_chessboard(color)]->color != color) valid_moves.emplace(move.rotate_to_chessboard(color));
        }
        return valid_moves;
    }
    void knight::print_piece(std::ostream& s) {
        s << 'N';
    }

    uint bishop::get_value() {
        return 3;
    }
    std::set<position> bishop::get_moves(chessboard& board) {
        std::set<position> valid_moves;
        std::array<position, 7> moves_up_left = { { {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x - 2), u(pos.y + 2)} ,
                                      {u(pos.x - 3), u(pos.y + 3)} , {u(pos.x - 4), u(pos.y + 4)} ,
                                      {u(pos.x - 5), u(pos.y + 5)} , {u(pos.x - 6), u(pos.y + 6)} ,
                                      {u(pos.x - 7), u(pos.y + 7)} } };
        std::array<position, 7>  moves_up_right = { { {u(pos.x + 1), u(pos.y + 1)} , {u(pos.x + 2), u(pos.y + 2)} ,
                                       {u(pos.x + 3), u(pos.y + 3)} , {u(pos.x + 4), u(pos.y + 4)} ,
                                       {u(pos.x + 5), u(pos.y + 5)} , {u(pos.x + 6), u(pos.y + 6)} ,
                                       {u(pos.x + 7), u(pos.y + 7)} } };
        std::array<position, 7>  moves_down_left = { { {u(pos.x - 1), u(pos.y - 1)} , {u(pos.x - 2), u(pos.y - 2)} ,
                                        {u(pos.x - 3), u(pos.y - 3)} , {u(pos.x - 4), u(pos.y - 4)} ,
                                        {u(pos.x - 5), u(pos.y - 5)} , {u(pos.x - 6), u(pos.y - 6)} ,
                                        {u(pos.x - 7), u(pos.y - 7)} } };
        std::array<position, 7>  moves_down_right = { { {u(pos.x + 1), u(pos.y - 1)} , {u(pos.x + 2), u(pos.y - 2)} ,
                                         {u(pos.x + 3), u(pos.y - 3)} , {u(pos.x + 4), u(pos.y - 4)} ,
                                         {u(pos.x + 5), u(pos.y - 5)} , {u(pos.x + 6), u(pos.y - 6)} ,
                                         {u(pos.x + 7), u(pos.y - 7)} } };

        for (const auto& moves : { moves_up_left, moves_up_right, moves_down_left, moves_down_right }) {
            for (const auto move : moves) {
                if (move.is_valid()) {
                    if (board.board[move.rotate_to_chessboard(color)]->color == color) break;
                    valid_moves.emplace(move.rotate_to_chessboard(color));
                    if (board.board[move.rotate_to_chessboard(color)]->color != 4) break;
                }
                else break;
            }
        }
        return valid_moves;
    }
    void bishop::print_piece(std::ostream& s) {
        s << 'B';
    }

    uint rock::get_value() {
        return 5;
    }
    std::set<position> rock::get_moves(chessboard& board) {
        std::set<position> valid_moves;
        std::array<position, 7> moves_up = { { {u(pos.x), u(pos.y + 1)}, { u(pos.x), u(pos.y + 2) },
                                    {u(pos.x), u(pos.y + 3)} , {u(pos.x), u(pos.y + 4)} ,
                                    {u(pos.x), u(pos.y + 5)} , {u(pos.x), u(pos.y + 6)} ,
                                    {u(pos.x), u(pos.y + 7)} } };
        std::array<position, 7> moves_down = { { {u(pos.x), u(pos.y - 1)} , {u(pos.x), u(pos.y - 2)} ,
                                    {u(pos.x), u(pos.y - 3)} , {u(pos.x), u(pos.y - 4)} ,
                                    {u(pos.x), u(pos.y - 5)} , {u(pos.x), u(pos.y - 6)} ,
                                    {u(pos.x), u(pos.y - 7)} } };
        std::array<position, 7> moves_left = { { {u(pos.x - 1), u(pos.y)}, { u(pos.x - 2), u(pos.y) },
                                    {u(pos.x - 3), u(pos.y)} , {u(pos.x - 4), u(pos.y)} ,
                                    {u(pos.x - 5), u(pos.y)} , {u(pos.x - 6), u(pos.y)} ,
                                    {u(pos.x - 7), u(pos.y)} } };
        std::array<position, 7> moves_right = { { {u(pos.x + 1), u(pos.y)} , {u(pos.x + 2), u(pos.y)} ,
                                    {u(pos.x + 3), u(pos.y)} , {u(pos.x + 4), u(pos.y)} ,
                                    {u(pos.x + 5), u(pos.y)} , {u(pos.x + 6), u(pos.y)} ,
                                    {u(pos.x + 7), u(pos.y)} } };
        //position* moves_array[] = { moves_up, moves_down, moves_left, moves_right };

        for (const auto& moves : { moves_up, moves_down, moves_left, moves_right }) {
            for (const auto move : moves) {
                if (move.is_valid()) {
                    if (board.board[move.rotate_to_chessboard(color)]->color == color) break;
                    valid_moves.emplace(move.rotate_to_chessboard(color));
                    if (board.board[move.rotate_to_chessboard(color)]->color != 4) break;
                }
                else break;
            }
        }
        return valid_moves;
    }
    void rock::print_piece(std::ostream& s) {
        s << 'R';
    }

    uint king::get_value() {
        return 3;
    }
    std::set<position> king::get_moves(chessboard& board) {
        std::set<position> valid_moves;
        position moves[8] = { {u(pos.x + 1), u(pos.y + 1)} , {u(pos.x + 1), u(pos.y - 1)} ,
                                 {u(pos.x - 1), u(pos.y + 1)} , {u(pos.x - 1), u(pos.y - 1)} ,
                                 {u(pos.x - 1), u(pos.y)} , {u(pos.x), u(pos.y - 1)} ,
                                 {u(pos.x + 1), u(pos.y)} , {u(pos.x), u(pos.y + 1)} };
        for (auto& move : moves) {
            if (move.is_valid() && board.board[move.rotate_to_chessboard(color)]->color != color) valid_moves.emplace(move.rotate_to_chessboard(color));
        }
        return valid_moves;
    }
    void king::print_piece(std::ostream& s) {
        s << 'K';
    }

    chessboard::chessboard() {
        turn = 0;
    };

    void chessboard::next_turn() {
        turn = (turn + 1) & 0x03;
    }

    bool chessboard::is_current_player_active() {
        return players[turn].active && !players[turn].out_of_time;
    }

    player& chessboard::current_player() {
        return players[turn];
    }

    uint chessboard::count_active() {
        uint count = 0;
        for (auto & p : players) {
            if (p.active && !p.out_of_time) count++;
        }
        return count;
    }

    //return true on success
    bool chessboard::make_move(action a) { //TODO: need to rotate back to env position
        if (board[a.from]->color < 4 && board[a.from]->color == turn) { //moving piece has to be player on turn
            std::set<position> moves = board[a.from]->get_moves(*this);
            if (moves.count(a.to)) {
                piece* p = board[a.from].get();
                //promoting pawn to rock on last row
                if (const pawn* p_pawn = dynamic_cast<pawn*>(p)) {
                    if (rotate_to_local(a.to, board[a.from]->color).y == 7) {
                        board[a.to] = std::make_unique<rock>(p_pawn->color, rotate_to_local(a.to, p_pawn->color));
                        board[a.from] = std::make_unique<piece>();
                        return true;
                    }
                }
                piece* p_taken = board[a.to].get();
                //promoting pawn to rock on last row
                if (const king* p_pawn = dynamic_cast<king*>(p_taken)) { //taking king means making other player inactive
                    players[board[a.from]->color].active = false;
                }
                board[a.from]->pos = rotate_to_local(a.to, board[a.from]->color);
                board[a.to] = std::move(board[a.from]);
                board[a.from] = std::make_unique<piece>();
                return true;
            }
        }
        return false;
    }

    std::map<position, piece*> chessboard::get_board() const {
        std::map<position, piece*> view;
        for (const auto& pos_piece_pair : board) {
            view[pos_piece_pair.first] = pos_piece_pair.second.get();  // Convert unique_ptr<piece> to piece*
        }
        return view;
    }

    ChaturajiEnv::ChaturajiEnv() {
        reset();
    }

    void ChaturajiEnv::reset() {
        _board.players[0] = player(0, 0);
        _board.players[1] = player(1, 0);
        _board.players[2] = player(2, 0);
        _board.players[3] = player(3, 0);

        _board.turn = 0;

        episode_length = 0;
        trajectory.clear();

        for (uint x = 0; x < 8; x++) {
            for (uint y = 0; y < 8; y++) {
                _board.board[{x, y}] = std::make_unique<piece>(4);
            }
        }
        //red
        _board.board[{0, 0}] = std::make_unique<rock>(0);
        _board.board[{1, 0}] = std::make_unique<knight>(0, position(1,0));
        _board.board[{2, 0}] = std::make_unique < bishop>(0, position(2,0));
        _board.board[{3, 0}] = std::make_unique < king>(0, position(3,0 ));
        _board.board[{0, 1}] = std::make_unique < pawn>(0, position(0,1 ), 0);
        _board.board[{1, 1}] = std::make_unique < pawn>(0, position( 1,1 ), 1);
        _board.board[{2, 1}] = std::make_unique < pawn>(0, position(2,1 ), 2);
        _board.board[{3, 1}] = std::make_unique < pawn>(0, position(3,1 ), 3);
        //blue
        _board.board[{0, 7}] = std::make_unique < rock>(1, position(0,0 ));
        _board.board[{0, 6}] = std::make_unique < knight>(1, position(1,0 ));
        _board.board[{0, 5}] = std::make_unique < bishop>(1, position(2,0 ));
        _board.board[{0, 4}] = std::make_unique < king>(1, position(3,0 ));
        _board.board[{1, 7}] = std::make_unique < pawn>(1, position(0,1 ), 0);
        _board.board[{1, 6}] = std::make_unique < pawn>(1, position(1,1 ), 1);
        _board.board[{1, 5}] = std::make_unique < pawn>(1, position(2,1 ), 2);
        _board.board[{1, 4}] = std::make_unique < pawn>(1, position(3,1 ), 3);
        //yellow
        _board.board[{7, 7}] = std::make_unique < rock>(2, position(0,0 ));
        _board.board[{6, 7}] = std::make_unique < knight>(2, position(1,0 ));
        _board.board[{5, 7}] = std::make_unique < bishop>(2, position(2,0 ));
        _board.board[{4, 7}] = std::make_unique < king>(2, position(3,0 ));
        _board.board[{7, 6}] = std::make_unique < pawn>(2, position(0,1 ), 1);
        _board.board[{6, 6}] = std::make_unique < pawn>(2, position(1,1 ), 2);
        _board.board[{5, 6}] = std::make_unique < pawn>(2, position(2,1 ), 3);
        _board.board[{4, 6}] = std::make_unique < pawn>(2, position(3,1 ), 4);
        //green
        _board.board[{7, 0}] = std::make_unique < rock>(3, position(0,0 ));
        _board.board[{7, 1}] = std::make_unique < knight>(3, position(1,0 ));
        _board.board[{7, 2}] = std::make_unique < bishop>(3, position(2,0 ));
        _board.board[{7, 3}] = std::make_unique < king>(3, position(3,0 ));
        _board.board[{6, 0}] = std::make_unique < pawn>(3, position(0,1 ), 0);
        _board.board[{6, 1}] = std::make_unique < pawn>(3, position(1,1 ), 1);
        _board.board[{6, 2}] = std::make_unique < pawn>(3, position(2,1 ), 2);
        _board.board[{6, 3}] = std::make_unique < pawn>(3, position(3,1 ), 3);

    }

    std::tuple<std::set<position>, std::set<position>, std::set<position>, std::set<position>, std::set<position>, std::set<position>, std::set<position>, std::set<position>> ChaturajiEnv::get_possible_actions() {
        std::set<position> moves_pawn1;
        std::set<position> moves_pawn2;
        std::set<position> moves_pawn3;
        std::set<position> moves_pawn4;
        std::set<position> moves_rock;
        std::set<position> moves_knight;
        std::set<position> moves_bishop;
        std::set<position> moves_king;
        for (uint x = 0; x < 8; x++) {
            for (uint y = 0; y < 8; y++) {
                if (_board.board[{x, y}]->color == _board.turn) {
                    auto *p = _board.board[{x, y}].get();
                    if (const pawn* p_pawn = dynamic_cast<pawn *>(p)) {
                        switch (p_pawn->index) {
                            case 0:
                                moves_pawn1 = p->get_moves(_board);
                                break;
                            case 1:
                                moves_pawn2 = p->get_moves(_board);
                                break;
                            case 2:
                                moves_pawn3 = p->get_moves(_board);
                                break;
                            case 3:
                                moves_pawn4 = p->get_moves(_board);
                                break;

                        }
                    }
                    else if (const rock* p_rock = dynamic_cast<rock*>(p)) {
                        moves_rock = p->get_moves(_board);
                    }
                    else if (const knight* p_knight = dynamic_cast<knight*>(p)) {
                        moves_knight = p->get_moves(_board);
                    }
                    else if (const bishop* p_bishop = dynamic_cast<bishop*>(p)) {
                        moves_bishop = p->get_moves(_board);
                    }
                    else if (const king* p_king = dynamic_cast<king*>(p)) {
                        moves_king = p->get_moves(_board);
                    }
                }
            }
        }
        return std::make_tuple(moves_pawn1, moves_pawn2, moves_pawn3, moves_pawn4,
                              moves_rock, moves_knight, moves_bishop, moves_king);
    }

    std::set<position> ChaturajiEnv::get_possible_actions_from(position from){
        return _board.board[from]->get_moves(_board);
    }

    std::tuple<bool, uint, bool, std::string> ChaturajiEnv::step(action a) { //state, reward, done, message
        uint reward = _board.board[a.to]->get_value();
        if (!_board.players[_board.board[a.to]->color].active || _board.players[_board.board[a.to]->color].out_of_time) {
            auto* p = _board.board[a.to].get();
            if (const king* p_pawn = dynamic_cast<king*>(p)) {
                reward = reward; //do nothing
            }
            else {
                reward = 0; //case where pawn which is taken is already not active
            }
        }
        bool done = false;
        if (!a.is_pass() && a.from.is_valid() && a.to.is_valid() && _board.is_current_player_active() && _board.make_move(a)) {
            trajectory.push_back(std::move(a));
            _board.current_player().score += reward;
        }
        else {
            _board.current_player().active = false;
            trajectory.push_back(action({ 0,0 }, { 0,0 }));
        }
        if (_board.count_active() < 2) {
            done = true;
            //TODO do counting of points for active player (taking all available kings)
        }
        //TODO implement auto resign - player is overhead with points more than all pieces at chessboard 
        _board.next_turn();
        episode_length++;
        while (!_board.current_player().active) {
            _board.next_turn();
            episode_length++;
            trajectory.push_back(action({ 0,0 }, { 0,0 }));
        }
        return std::make_tuple(true, reward, done, "");
    }

    void ChaturajiEnv::render(std::ostream& s) {
        for (uint x = 0; x < 8; x++) {
            for (uint y = 0; y < 8; y++) {
                _board.board[{x, y}]->print_piece(s);
                //std::cout << (int) _board.board[{x, y}]->get_value();
                print_color(_board.board[{x, y}]->get_color());
                s << ' ';
            }
            s << '\n' << '\n';
        }
        for (uint p_i = 0; p_i < 4; p_i++) {
            s << "player ";
            print_color(_board.players[p_i].color, s);
            s << ": " << (int) _board.players[p_i].score << '\n';
        }
        s << "Turn player: "; 
        print_color(_board.current_player().color);
        s << '\n';
    }

    void ChaturajiEnv::close(std::ostream& s) {
        s << "Closing environment." << std::endl;
    }

PYBIND11_MODULE(chaturajienv, m) {
    // Expose the position struct to Python
    py::class_<position>(m, "Position")
        .def(py::init<>())  // Default constructor
        .def_readwrite("x", &position::x)
        .def_readwrite("y", &position::y);

    py::class_<action>(m, "Action")
        .def(py::init<>())  // Default constructor
        .def(py::init<position, position>(), "Initialize action",
            py::arg("from") = position(), py::arg("to") = position())
        .def_readwrite("from", &action::from)
        .def_readwrite("to", &action::to);

    py::class_<piece>(m, "Piece")
        .def(py::init<>())
        .def(py::init<uint, position>())
        .def("get_color", &piece::get_color)
        .def("get_value", &piece::get_value)
        .def("get_moves", &piece::get_moves);

    py::class_<pawn, piece>(m, "Pawn")
        .def(py::init<uint, position, unsigned short>())
        .def("get_value", &pawn::get_value)
        .def("get_moves", &pawn::get_moves);

    py::class_<knight, piece>(m, "Knight")
        .def(py::init<uint, position>())
        .def("get_value", &knight::get_value)
        .def("get_moves", &knight::get_moves);

    py::class_<bishop, piece>(m, "Bishop")
        .def(py::init<uint, position>())
        .def("get_value", &bishop::get_value)
        .def("get_moves", &bishop::get_moves);

    py::class_<rock, piece>(m, "Rock")
        .def(py::init<uint, position>())
        .def("get_value", &rock::get_value)
        .def("get_moves", &rock::get_moves);

    py::class_<king, piece>(m, "King")
        .def(py::init<uint, position>())
        .def("get_value", &king::get_value)
        .def("get_moves", &king::get_moves);

    py::class_<chessboard>(m, "Chessboard")
        .def("get_board", &chessboard::get_board, py::return_value_policy::copy);

    // Expose the ChaturajiEnv class and its get_possible_actions() method
    py::class_<ChaturajiEnv>(m, "ChaturajiEnv")
        .def(py::init<>())  // Default constructor
        .def("reset", &ChaturajiEnv::reset,
            "Reset the board and players to initial starting position")
        .def("step", &ChaturajiEnv::step,
            "Given position from and position to, make a step if correct, if incorrect player is disqualified")
        .def("render", &ChaturajiEnv::render_p, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
            "Render board in human readable way")
        .def("get_possible_actions", &ChaturajiEnv::get_possible_actions,
            "Return a tuple of sets of possible actions (positions) for current player")
        .def("get_possible_actions_from", &ChaturajiEnv::get_possible_actions_from,
            "Return a set of possible actions (positions) from given position")
        .def("get_board", &ChaturajiEnv::get_board, py::return_value_policy::reference_internal);
}

/*int main() {
    ChaturajiEnv env = ChaturajiEnv();
    for (int i = 0; i < 100; i++) {
        env.render();
        uint x_f, y_f, x_t, y_t;
        std::cout << "From: ";
        std::cin >> x_f >> y_f;
        std::cout << "Moves: ";
        for (auto const& move : env.get_possible_actions_from({ x_f, y_f })) {
            std::cout << move << " ";
        }
        std::cout << std::endl;
        std::cout << "To: ";
        std::cin >> x_t >> y_t;

        env.step(action({ x_f, y_f }, { x_t, y_t }));
    }
}*/
