// ============================================================================
//  Simple Bitboard Chess Engine  (C++17, single translation unit)
//
//  - Bitboard board representation (U64 per piece type / color)
//  - Magic bitboards for sliding attacks (magics generated at startup with a
//    fixed-seed PRNG, so no magic constants are hard-coded)
//  - Fully legal move generation (pseudo-legal + make/test legality filter)
//  - Perft validation suite (startpos + Kiwipete)
//  - Alpha-beta search: iterative deepening, transposition table (Zobrist),
//    quiescence search, MVV-LVA / killer / history move ordering
//  - Evaluation: material + piece-square tables (pluggable via eval())
//  - UCI protocol
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <thread>
#include <mutex>
#endif

typedef uint64_t U64;
typedef int16_t  I16;

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------
static inline int popcount(U64 b) { return __builtin_popcountll(b); }
static inline int lsb(U64 b)      { return __builtin_ctzll(b); }
static inline int poplsb(U64& b)  { int s = lsb(b); b &= b - 1; return s; }
static inline U64 bit(int sq)     { return 1ULL << sq; }

// Squares: 0 = a1 ... 63 = h8.  file = sq & 7, rank = sq >> 3.
enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };
enum { EMPTY = -1 };

static const U64 FILE_A     = 0x0101010101010101ULL;
static const U64 FILE_H     = 0x8080808080808080ULL;
static const U64 NOT_FILE_A = ~FILE_A;
static const U64 NOT_FILE_H = ~FILE_H;

// Castling rights bitmask
enum { CR_WK = 1, CR_WQ = 2, CR_BK = 4, CR_BQ = 8 };

// ---------------------------------------------------------------------------
// PRNG (xorshift64*) -- fixed seed => deterministic magic/zobrist generation
// ---------------------------------------------------------------------------
static U64 rngState = 0x9E3779B97F4A7C15ULL;
static U64 rng() {
    rngState ^= rngState >> 12;
    rngState ^= rngState << 25;
    rngState ^= rngState >> 27;
    return rngState * 2685821657736338717ULL;
}
static U64 rngFewBits() { return rng() & rng() & rng(); }

// ---------------------------------------------------------------------------
// Attack tables
// ---------------------------------------------------------------------------
static U64 PAWN_ATTACKS[2][64];
static U64 KNIGHT_ATTACKS[64];
static U64 KING_ATTACKS[64];

// Magic bitboard state (fancy, fixed shift: 12 bits rook / 9 bits bishop)
static U64 ROOK_MASK[64],   BISHOP_MASK[64];
static U64 ROOK_MAGIC[64],  BISHOP_MAGIC[64];
static U64 ROOK_TABLE[64][4096];   // 64 * 4096 * 8 = 2 MB
static U64 BISHOP_TABLE[64][512];  // 64 *  512 * 8 = 256 KB

// Sliding attacks on the fly: walk each ray, include the first blocker, stop there.
static U64 rayAttacks(int sq, const int dirs[4][2], int ndirs, U64 occ) {
    U64 att = 0;
    int f0 = sq & 7, r0 = sq >> 3;
    for (int d = 0; d < ndirs; d++) {
        int f = f0 + dirs[d][0], r = r0 + dirs[d][1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int s = r * 8 + f;
            att |= bit(s);
            if (occ & bit(s)) break;
            f += dirs[d][0];
            r += dirs[d][1];
        }
    }
    return att;
}

// Magic "relevant occupancy" mask: all squares on the rays except the last
// square of each ray (the board-edge square, which can never change attacks).
static U64 rayMask(int sq, const int dirs[4][2], int ndirs) {
    U64 att = 0;
    int f0 = sq & 7, r0 = sq >> 3;
    for (int d = 0; d < ndirs; d++) {
        int f = f0 + dirs[d][0], r = r0 + dirs[d][1];
        int lastSq = -1;
        U64 dirAtt = 0;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            lastSq = r * 8 + f;
            dirAtt |= bit(lastSq);
            f += dirs[d][0];
            r += dirs[d][1];
        }
        if (lastSq >= 0) dirAtt &= ~bit(lastSq);
        att |= dirAtt;
    }
    return att;
}

static const int ROOK_DIRS[4][2]   = {{1,0},{-1,0},{0,1},{0,-1}};
static const int BISHOP_DIRS[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};

static U64 rookMaskSlow(int sq)     { return rayMask(sq, ROOK_DIRS, 4); }
static U64 bishopMaskSlow(int sq)   { return rayMask(sq, BISHOP_DIRS, 4); }
static U64 rookAttacksSlow(int sq, U64 occ)   { return rayAttacks(sq, ROOK_DIRS, 4, occ); }
static U64 bishopAttacksSlow(int sq, U64 occ) { return rayAttacks(sq, BISHOP_DIRS, 4, occ); }

static U64 findMagic(int sq, U64 mask, int shift, bool bishop) {
    int bits = popcount(mask);
    int n = 1 << bits;
    std::vector<U64> subsets(n), attacks(n);
    U64 b = 0;
    for (int i = 0; i < n; i++) {
        subsets[i] = b;
        attacks[i] = bishop ? bishopAttacksSlow(sq, b) : rookAttacksSlow(sq, b);
        b = (b - mask) & mask;   // carry-ripper subset enumeration
    }
    int tableSize = 1 << (64 - shift);
    std::vector<U64> used(tableSize);
    std::vector<char> seen(tableSize);
    for (long long attempt = 0; attempt < 1000000000LL; attempt++) {
        U64 magic = rngFewBits();
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;
        std::fill(seen.begin(), seen.end(), 0);
        bool ok = true;
        for (int i = 0; i < n; i++) {
            int idx = (int)((subsets[i] * magic) >> shift);
            if (!seen[idx]) { seen[idx] = 1; used[idx] = attacks[i]; }
            else if (used[idx] != attacks[i]) { ok = false; break; }
        }
        if (ok) return magic;
    }
    fprintf(stderr, "findMagic failed for square %d\n", sq);
    exit(1);
}

static void initAttacks() {
    // Pawns
    for (int s = 0; s < 64; s++) {
        U64 b = bit(s);
        PAWN_ATTACKS[WHITE][s] = ((b & NOT_FILE_A) << 7) | ((b & NOT_FILE_H) << 9);
        PAWN_ATTACKS[BLACK][s] = ((b & NOT_FILE_A) >> 9) | ((b & NOT_FILE_H) >> 7);
    }
    // Knights
    static const int KJ[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    for (int s = 0; s < 64; s++) {
        U64 att = 0;
        int f = s & 7, r = s >> 3;
        for (int i = 0; i < 8; i++) {
            int nf = f + KJ[i][0], nr = r + KJ[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) att |= bit(nr * 8 + nf);
        }
        KNIGHT_ATTACKS[s] = att;
    }
    // Kings
    for (int s = 0; s < 64; s++) {
        U64 att = 0;
        int f = s & 7, r = s >> 3;
        for (int df = -1; df <= 1; df++) for (int dr = -1; dr <= 1; dr++) {
            if (!df && !dr) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) att |= bit(nr * 8 + nf);
        }
        KING_ATTACKS[s] = att;
    }
    // Magic bitboards: generate magics, then fill attack tables.
    for (int s = 0; s < 64; s++) {
        ROOK_MASK[s]   = rookMaskSlow(s);
        BISHOP_MASK[s] = bishopMaskSlow(s);
        ROOK_MAGIC[s]   = findMagic(s, ROOK_MASK[s], 52, false);
        BISHOP_MAGIC[s] = findMagic(s, BISHOP_MASK[s], 55, true);
        U64 b = 0;   // rook subsets of the (already relevant-only) mask
        do {
            int ri = (int)((b * ROOK_MAGIC[s]) >> 52);
            ROOK_TABLE[s][ri] = rookAttacksSlow(s, b);
            b = (b - ROOK_MASK[s]) & ROOK_MASK[s];
        } while (b);
        b = 0;       // bishop subsets
        do {
            int bi = (int)((b * BISHOP_MAGIC[s]) >> 55);
            BISHOP_TABLE[s][bi] = bishopAttacksSlow(s, b);
            b = (b - BISHOP_MASK[s]) & BISHOP_MASK[s];
        } while (b);
    }
}

static inline U64 rookAttacks(int sq, U64 occ) {
    return ROOK_TABLE[sq][((occ & ROOK_MASK[sq]) * ROOK_MAGIC[sq]) >> 52];
}
static inline U64 bishopAttacks(int sq, U64 occ) {
    return BISHOP_TABLE[sq][((occ & BISHOP_MASK[sq]) * BISHOP_MAGIC[sq]) >> 55];
}
static inline U64 queenAttacks(int sq, U64 occ) {
    return rookAttacks(sq, occ) | bishopAttacks(sq, occ);
}

// ---------------------------------------------------------------------------
// Zobrist keys
// ---------------------------------------------------------------------------
static U64 Z_PIECE[2][6][64];
static U64 Z_CASTLE[16];
static U64 Z_EP[9];        // index 8 = no en passant
static U64 Z_SIDE;
static bool zobristReady = false;

static void initZobrist() {
    for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++) for (int s = 0; s < 64; s++)
        Z_PIECE[c][p][s] = rng();
    for (int i = 0; i < 16; i++) Z_CASTLE[i] = rng();
    for (int i = 0; i < 9;  i++) Z_EP[i] = rng();
    Z_SIDE = rng();
    zobristReady = true;
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------
struct Undo {
    int captured;       // piece type or EMPTY
    int epSquare;       // -1 if none
    int castling;
    int halfmove;
    U64 hash;
};

static const int HIST_MAX = 2048;   // position-key history for repetition detection

struct Position {
    U64 pieces[2][6];   // pieces[color][type]
    U64 occ[3];         // occ[WHITE], occ[BLACK], occ[2] = all
    int  board[64];     // piece type or EMPTY
    int  colorOf[64];   // WHITE/BLACK or -1
    int  side;          // side to move
    int  castling;      // CR_* bitmask
    int  epSquare;      // -1 if none
    int  halfmove;
    int  fullmove;
    U64  hash;
    U64  keyHist[HIST_MAX];   // Zobrist keys, one per played position
    int  histPly;

    void clear() {
        for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++) pieces[c][p] = 0;
        occ[0] = occ[1] = occ[2] = 0;
        for (int s = 0; s < 64; s++) { board[s] = EMPTY; colorOf[s] = -1; }
        side = WHITE; castling = 0; epSquare = -1; halfmove = 0; fullmove = 1; hash = 0;
        histPly = 0;
    }

    void putPiece(int c, int p, int s) {
        pieces[c][p] |= bit(s);
        occ[c] |= bit(s); occ[2] |= bit(s);
        board[s] = p; colorOf[s] = c;
    }

    int kingSq(int c) const { return lsb(pieces[c][KING]); }

    // Standard board queries (piece-at-square, colour-at-square, occupancy)
    int  pieceAt(int s)    const { return board[s]; }              // type or EMPTY
    int  colorAt(int s)    const { return colorOf[s]; }            // WHITE/BLACK or -1
    bool isOccupied(int s) const { return (occ[2] >> s) & 1; }

    U64 computeHash() const {
        U64 h = 0;
        for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++) {
            U64 b = pieces[c][p];
            while (b) h ^= Z_PIECE[c][p][poplsb(b)];
        }
        h ^= Z_CASTLE[castling];
        h ^= Z_EP[epSquare >= 0 ? (epSquare & 7) : 8];
        if (side == BLACK) h ^= Z_SIDE;
        return h;
    }

    bool isAttacked(int sq, int by) const {
        U64 all = occ[2];
        if (PAWN_ATTACKS[by ^ 1][sq] & pieces[by][PAWN]) return true;
        if (KNIGHT_ATTACKS[sq] & pieces[by][KNIGHT]) return true;
        if (KING_ATTACKS[sq] & pieces[by][KING]) return true;
        if (bishopAttacks(sq, all) & (pieces[by][BISHOP] | pieces[by][QUEEN])) return true;
        if (rookAttacks(sq, all) & (pieces[by][ROOK] | pieces[by][QUEEN])) return true;
        return false;
    }

    bool inCheck(int c) const { return isAttacked(kingSq(c), c ^ 1); }
};

// ---------------------------------------------------------------------------
// Moves: packed into 32 bits
//   bits  0- 5 : from square
//   bits  6-11 : to square
//   bits 12-14 : promotion piece (KNIGHT..QUEEN), 0 = none
//   bit  15    : capture
//   bits 16-17 : flags (1 = double push, 2 = en passant, 3 = castle)
//   bits 24-31 : ordering score
// ---------------------------------------------------------------------------
typedef uint32_t Move;
static const Move MOVE_NONE = 0;

static inline int  mvFrom(Move m)  { return (int)(m & 63); }
static inline int  mvTo(Move m)    { return (int)((m >> 6) & 63); }
static inline int  mvPromo(Move m) { return (int)((m >> 12) & 7); }
static inline bool mvIsCap(Move m) { return (m >> 15) & 1; }
static inline int  mvFlags(Move m) { return (int)((m >> 16) & 3); }

static inline Move makeMove(int from, int to, int promo = 0, bool cap = false, int flags = 0) {
    return (Move)(from | (to << 6) | (promo << 12) | ((cap ? 1u : 0u) << 15) | (flags << 16));
}

static std::string moveToStr(Move m) {
    if (!m) return "0000";
    std::string s;
    s += (char)('a' + (mvFrom(m) & 7));
    s += (char)('1' + (mvFrom(m) >> 3));
    s += (char)('a' + (mvTo(m) & 7));
    s += (char)('1' + (mvTo(m) >> 3));
    if (mvPromo(m)) s += "nbrq"[mvPromo(m) - KNIGHT];
    return s;
}

// ---------------------------------------------------------------------------
// Make / unmake move
// ---------------------------------------------------------------------------
// Castling rights cleared when a piece moves FROM or is captured ON a square.
static int CR_CLEAR[64];
static void initCastlingClear() {
    for (int s = 0; s < 64; s++) CR_CLEAR[s] = 15;
    CR_CLEAR[0]  = 15 & ~CR_WQ;              // a1
    CR_CLEAR[7]  = 15 & ~CR_WK;              // h1
    CR_CLEAR[4]  = 15 & ~(CR_WK | CR_WQ);    // e1
    CR_CLEAR[56] = 15 & ~CR_BQ;              // a8
    CR_CLEAR[63] = 15 & ~CR_BK;              // h8
    CR_CLEAR[60] = 15 & ~(CR_BK | CR_BQ);    // e8
}

static inline void removePiece(Position& pos, int c, int p, int s) {
    pos.pieces[c][p] &= ~bit(s);
    pos.occ[c] &= ~bit(s);
    pos.occ[2] &= ~bit(s);
    pos.board[s] = EMPTY;
    pos.colorOf[s] = -1;
    pos.hash ^= Z_PIECE[c][p][s];
}
static inline void addPiece(Position& pos, int c, int p, int s) {
    pos.pieces[c][p] |= bit(s);
    pos.occ[c] |= bit(s);
    pos.occ[2] |= bit(s);
    pos.board[s] = p;
    pos.colorOf[s] = c;
    pos.hash ^= Z_PIECE[c][p][s];
}
static inline void movePiece(Position& pos, int c, int p, int from, int to) {
    U64 bb = bit(from) | bit(to);
    pos.pieces[c][p] ^= bb;
    pos.occ[c] ^= bb;
    pos.occ[2] ^= bb;
    pos.board[from] = EMPTY;
    pos.colorOf[from] = -1;
    pos.board[to] = p;
    pos.colorOf[to] = c;
    pos.hash ^= Z_PIECE[c][p][from] ^ Z_PIECE[c][p][to];
}

static void makeMove(Position& pos, Move m, Undo& u) {
    int from = mvFrom(m), to = mvTo(m);
    int us = pos.side, them = us ^ 1;
    int flags = mvFlags(m);
    int promo = mvPromo(m);

    u.captured = pos.board[to];
    u.epSquare = pos.epSquare;
    u.castling = pos.castling;
    u.halfmove = pos.halfmove;
    u.hash = pos.hash;

    // XOR out old en passant + castling state
    pos.hash ^= Z_EP[pos.epSquare >= 0 ? (pos.epSquare & 7) : 8];
    pos.hash ^= Z_CASTLE[pos.castling];

    int piece = pos.board[from];
    pos.halfmove++;

    // Normal capture removal FIRST (before placing mover on 'to')
    if (mvIsCap(m) && flags != 2) {
        removePiece(pos, them, u.captured, to);
        pos.halfmove = 0;
    }

    if (flags == 3) {
        // Castling: king from e1/e8
        if (to == 6)       { movePiece(pos, us, KING, 4, 6);  movePiece(pos, us, ROOK, 7, 5); }
        else if (to == 2)  { movePiece(pos, us, KING, 4, 2);  movePiece(pos, us, ROOK, 0, 3); }
        else if (to == 62) { movePiece(pos, us, KING, 60, 62); movePiece(pos, us, ROOK, 63, 61); }
        else               { movePiece(pos, us, KING, 60, 58); movePiece(pos, us, ROOK, 56, 59); }
        pos.halfmove = 0;
    } else if (flags == 2) {
        // En passant capture
        int capSq = (us == WHITE) ? to - 8 : to + 8;
        removePiece(pos, them, PAWN, capSq);
        movePiece(pos, us, PAWN, from, to);
        pos.halfmove = 0;
    } else if (promo) {
        removePiece(pos, us, PAWN, from);
        addPiece(pos, us, promo, to);
        pos.halfmove = 0;
    } else {
        movePiece(pos, us, piece, from, to);
        if (piece == PAWN) pos.halfmove = 0;
    }

    // Double push -> set EP square
    pos.epSquare = -1;
    if (flags == 1) pos.epSquare = (us == WHITE) ? from + 8 : from - 8;

    // Update castling rights
    pos.castling &= CR_CLEAR[from] & CR_CLEAR[to];

    // XOR in new en passant + castling state
    pos.hash ^= Z_EP[pos.epSquare >= 0 ? (pos.epSquare & 7) : 8];
    pos.hash ^= Z_CASTLE[pos.castling];

    pos.side = them;
    pos.hash ^= Z_SIDE;
    if (us == BLACK) pos.fullmove++;

    // Record the new position key (for repetition detection)
    if (pos.histPly >= HIST_MAX) {   // keep the window; even shift preserves side-to-move parity
        memmove(pos.keyHist, pos.keyHist + HIST_MAX / 2, (HIST_MAX / 2) * sizeof(U64));
        pos.histPly = HIST_MAX / 2;
    }
    pos.keyHist[pos.histPly++] = pos.hash;
}

static void unmakeMove(Position& pos, Move m, const Undo& u) {
    int from = mvFrom(m), to = mvTo(m);
    int flags = mvFlags(m);
    int promo = mvPromo(m);
    pos.side ^= 1;
    int us = pos.side, them = us ^ 1;
    if (us == BLACK) pos.fullmove--;   // undo the increment done by Black's move

    if (flags == 3) {
        if (to == 6)       { movePiece(pos, us, KING, 6, 4);  movePiece(pos, us, ROOK, 5, 7); }
        else if (to == 2)  { movePiece(pos, us, KING, 2, 4);  movePiece(pos, us, ROOK, 3, 0); }
        else if (to == 62) { movePiece(pos, us, KING, 62, 60); movePiece(pos, us, ROOK, 61, 63); }
        else               { movePiece(pos, us, KING, 58, 60); movePiece(pos, us, ROOK, 59, 56); }
    } else if (flags == 2) {
        movePiece(pos, us, PAWN, to, from);
        int capSq = (us == WHITE) ? to - 8 : to + 8;
        addPiece(pos, them, PAWN, capSq);
    } else if (promo) {
        removePiece(pos, us, promo, to);
        addPiece(pos, us, PAWN, from);
    } else {
        movePiece(pos, us, pos.board[to], to, from);
    }

    if (mvIsCap(m) && flags != 2) addPiece(pos, them, u.captured, to);

    pos.castling = u.castling;
    pos.epSquare = u.epSquare;
    pos.halfmove = u.halfmove;
    pos.hash = u.hash;
    if (pos.histPly > 0) pos.histPly--;
}

// True if the current position has occurred before within the current
// irreversible-move segment (i.e. an actual threefold-rule repetition).
static bool isRepetition(const Position& pos) {
    int oldest = pos.histPly - 1 - pos.halfmove;   // first index inside current segment
    if (oldest < 0) oldest = 0;
    for (int i = pos.histPly - 3; i >= oldest; i -= 2)
        if (pos.keyHist[i] == pos.hash) return true;
    return false;
}

// ---------------------------------------------------------------------------
// FEN parsing
// ---------------------------------------------------------------------------
static int charToPiece(char ch) {
    switch (ch) {
        case 'p': case 'P': return PAWN;
        case 'n': case 'N': return KNIGHT;
        case 'b': case 'B': return BISHOP;
        case 'r': case 'R': return ROOK;
        case 'q': case 'Q': return QUEEN;
        case 'k': case 'K': return KING;
    }
    return -1;
}

static bool setFEN(Position& pos, const std::string& fen) {
    pos.clear();
    std::istringstream ss(fen);
    std::string boardStr, sideStr, castleStr, epStr;
    ss >> boardStr >> sideStr >> castleStr >> epStr;
    if (boardStr.empty()) return false;

    int sq = 56;  // start at a8
    for (size_t i = 0; i < boardStr.size(); i++) {
        char ch = boardStr[i];
        if (ch == '/') { sq -= 16; continue; }
        if (ch >= '1' && ch <= '8') { sq += ch - '0'; continue; }
        int p = charToPiece(ch);
        if (p < 0 || sq < 0 || sq > 63) return false;
        int c = (ch >= 'a' && ch <= 'z') ? BLACK : WHITE;
        pos.putPiece(c, p, sq);
        sq++;
    }

    pos.side = (sideStr == "b") ? BLACK : WHITE;

    pos.castling = 0;
    if (castleStr.find('K') != std::string::npos) pos.castling |= CR_WK;
    if (castleStr.find('Q') != std::string::npos) pos.castling |= CR_WQ;
    if (castleStr.find('k') != std::string::npos) pos.castling |= CR_BK;
    if (castleStr.find('q') != std::string::npos) pos.castling |= CR_BQ;

    pos.epSquare = -1;
    if (epStr.size() == 2 && epStr[0] >= 'a' && epStr[0] <= 'h' &&
        epStr[1] >= '1' && epStr[1] <= '8')
        pos.epSquare = (epStr[1] - '1') * 8 + (epStr[0] - 'a');

    if (!(ss >> pos.halfmove)) pos.halfmove = 0;
    if (!(ss >> pos.fullmove)) pos.fullmove = 1;

    if (zobristReady) {
        pos.hash = pos.computeHash();
        pos.keyHist[0] = pos.hash;
        pos.histPly = 1;
    }
    return true;
}

static const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static void printBoard(const Position& pos) {
    const char* chars = "PNBRQK";
    std::ostringstream out;
    for (int r = 7; r >= 0; r--) {
        out << (r + 1) << "  ";
        for (int f = 0; f < 8; f++) {
            int s = r * 8 + f;
            char ch = '.';
            if (pos.board[s] != EMPTY) {
                ch = chars[pos.board[s]];
                if (pos.colorOf[s] == BLACK) ch = (char)tolower(ch);
            }
            out << ch << ' ';
        }
        out << "\n";
    }
    out << "   a b c d e f g h\n";
    out << "Side: " << (pos.side == WHITE ? "w" : "b")
        << "  Castling: "
        << ((pos.castling & CR_WK) ? 'K' : '-')
        << ((pos.castling & CR_WQ) ? 'Q' : '-')
        << ((pos.castling & CR_BK) ? 'k' : '-')
        << ((pos.castling & CR_BQ) ? 'q' : '-')
        << "  EP: " << pos.epSquare;
    std::cout << out.str() << std::endl;
}

// ---------------------------------------------------------------------------
// Move generation (pseudo-legal; legality filtered by make/test)
// ---------------------------------------------------------------------------
static const int MAX_MOVES = 256;

struct MoveList {
    Move moves[MAX_MOVES];
    int  scores[MAX_MOVES];   // move-ordering scores (kept out of the move word)
    int  n = 0;
    void add(Move m) { if (n < MAX_MOVES) { moves[n] = m; scores[n] = 0; n++; } }
};

static void genPawnMoves(const Position& pos, MoveList& ml, int us, bool capturesOnly) {
    U64 pawns = pos.pieces[us][PAWN];
    U64 empty = ~pos.occ[2];
    U64 enemies = pos.occ[us ^ 1];
    int up = (us == WHITE) ? 8 : -8;
    U64 promoRank = (us == WHITE) ? 0xFF00000000000000ULL : 0x00000000000000FFULL;
    U64 dblRank   = (us == WHITE) ? 0x000000000000FF00ULL : 0x00FF000000000000ULL;

    U64 b = pawns;
    while (b) {
        int from = poplsb(b);
        U64 fromBit = bit(from);

        // Captures (+ promotions)
        U64 caps = PAWN_ATTACKS[us][from] & enemies;
        while (caps) {
            int to = poplsb(caps);
            if (bit(to) & promoRank) {
                ml.add(makeMove(from, to, QUEEN,  true));
                ml.add(makeMove(from, to, ROOK,   true));
                ml.add(makeMove(from, to, BISHOP, true));
                ml.add(makeMove(from, to, KNIGHT, true));
            } else {
                ml.add(makeMove(from, to, 0, true));
            }
        }
        // En passant
        if (pos.epSquare >= 0 && (PAWN_ATTACKS[us][from] & bit(pos.epSquare)))
            ml.add(makeMove(from, pos.epSquare, 0, true, 2));

        if (!capturesOnly) {
            int to = from + up;
            if (to >= 0 && to < 64 && (empty & bit(to))) {
                if (bit(to) & promoRank) {
                    ml.add(makeMove(from, to, QUEEN));
                    ml.add(makeMove(from, to, ROOK));
                    ml.add(makeMove(from, to, BISHOP));
                    ml.add(makeMove(from, to, KNIGHT));
                } else {
                    ml.add(makeMove(from, to));
                    int to2 = from + 2 * up;
                    if ((fromBit & dblRank) && (empty & bit(to2)))
                        ml.add(makeMove(from, to2, 0, false, 1));
                }
            }
        }
    }
}

static void genCastling(const Position& pos, MoveList& ml, int us) {
    if (us == WHITE) {
        if ((pos.castling & CR_WK) && !(pos.occ[2] & 0x60ULL)) {           // f1,g1 empty
            if (!pos.isAttacked(4, BLACK) && !pos.isAttacked(5, BLACK) && !pos.isAttacked(6, BLACK))
                ml.add(makeMove(4, 6, 0, false, 3));
        }
        if ((pos.castling & CR_WQ) && !(pos.occ[2] & 0x0EULL)) {           // b1,c1,d1 empty
            if (!pos.isAttacked(4, BLACK) && !pos.isAttacked(3, BLACK) && !pos.isAttacked(2, BLACK))
                ml.add(makeMove(4, 2, 0, false, 3));
        }
    } else {
        if ((pos.castling & CR_BK) && !(pos.occ[2] & 0x6000000000000000ULL)) {
            if (!pos.isAttacked(60, WHITE) && !pos.isAttacked(61, WHITE) && !pos.isAttacked(62, WHITE))
                ml.add(makeMove(60, 62, 0, false, 3));
        }
        if ((pos.castling & CR_BQ) && !(pos.occ[2] & 0x0E00000000000000ULL)) {
            if (!pos.isAttacked(60, WHITE) && !pos.isAttacked(59, WHITE) && !pos.isAttacked(58, WHITE))
                ml.add(makeMove(60, 58, 0, false, 3));
        }
    }
}

static void genPseudo(const Position& pos, MoveList& ml, bool capturesOnly) {
    int us = pos.side;
    U64 own = pos.occ[us];
    U64 enemies = pos.occ[us ^ 1];

    genPawnMoves(pos, ml, us, capturesOnly);

    // Knights
    U64 b = pos.pieces[us][KNIGHT];
    while (b) {
        int from = poplsb(b);
        U64 att = KNIGHT_ATTACKS[from] & ~own;
        while (att) {
            int to = poplsb(att);
            bool cap = (enemies >> to) & 1;
            if (capturesOnly && !cap) continue;
            ml.add(makeMove(from, to, 0, cap));
        }
    }
    // Bishops + queens (diagonal)
    b = pos.pieces[us][BISHOP] | pos.pieces[us][QUEEN];
    while (b) {
        int from = poplsb(b);
        U64 att = bishopAttacks(from, pos.occ[2]) & ~own;
        while (att) {
            int to = poplsb(att);
            bool cap = (enemies >> to) & 1;
            if (capturesOnly && !cap) continue;
            ml.add(makeMove(from, to, 0, cap));
        }
    }
    // Rooks + queens (straight)
    b = pos.pieces[us][ROOK] | pos.pieces[us][QUEEN];
    while (b) {
        int from = poplsb(b);
        U64 att = rookAttacks(from, pos.occ[2]) & ~own;
        while (att) {
            int to = poplsb(att);
            bool cap = (enemies >> to) & 1;
            if (capturesOnly && !cap) continue;
            ml.add(makeMove(from, to, 0, cap));
        }
    }
    // King
    {
        int from = pos.kingSq(us);
        U64 att = KING_ATTACKS[from] & ~own;
        while (att) {
            int to = poplsb(att);
            bool cap = (enemies >> to) & 1;
            if (capturesOnly && !cap) continue;
            ml.add(makeMove(from, to, 0, cap));
        }
    }
    if (!capturesOnly) genCastling(pos, ml, us);
}

static int genLegal(Position& pos, MoveList& out) {
    MoveList pseudo;
    genPseudo(pos, pseudo, false);
    int us = pos.side;
    out.n = 0;
    for (int i = 0; i < pseudo.n; i++) {
        Undo u;
        makeMove(pos, pseudo.moves[i], u);
        if (!pos.inCheck(us)) out.add(pseudo.moves[i]);
        unmakeMove(pos, pseudo.moves[i], u);
    }
    return out.n;
}

// ---------------------------------------------------------------------------
// Perft
// ---------------------------------------------------------------------------
static U64 perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    genPseudo(pos, ml, false);
    int us = pos.side;
    U64 nodes = 0;
    for (int i = 0; i < ml.n; i++) {
        Undo u;
        makeMove(pos, ml.moves[i], u);
        if (!pos.inCheck(us)) nodes += perft(pos, depth - 1);
        unmakeMove(pos, ml.moves[i], u);
    }
    return nodes;
}

static void perftDivide(Position& pos, int depth) {
    MoveList ml;
    genPseudo(pos, ml, false);
    int us = pos.side;
    U64 total = 0;
    for (int i = 0; i < ml.n; i++) {
        Undo u;
        makeMove(pos, ml.moves[i], u);
        if (!pos.inCheck(us)) {
            U64 cnt = perft(pos, depth - 1);
            total += cnt;
            std::cout << moveToStr(ml.moves[i]) << ": " << cnt << "\n";
        }
        unmakeMove(pos, ml.moves[i], u);
    }
    std::cout << "Total: " << total << std::endl;
}

static bool runPerftSuite() {
    struct Test { const char* name; const char* fen; int depth; U64 expected; };
    static const U64 startExp[6] = {20, 400, 8902, 197281, 4865609, 119060324};
    static const U64 kiwiExp[4]  = {48, 2039, 97862, 4085603};
    const char* kiwiFen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    bool allPass = true;
    Position pos;

    std::cout << "=== Perft: standard starting position ===" << std::endl;
    for (int d = 1; d <= 6; d++) {
        setFEN(pos, START_FEN);
        auto t0 = std::chrono::steady_clock::now();
        U64 got = perft(pos, d);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bool pass = (got == startExp[d - 1]);
        if (!pass) allPass = false;
        std::cout << "depth " << d << ": " << got << "  expected " << startExp[d - 1]
                  << "  " << (pass ? "PASS" : "FAIL")
                  << "  (" << (long long)ms << " ms, " << (ms > 0 ? got / ms / 1000.0 : 0.0)
                  << " Mnps)\n";
    }

    std::cout << "=== Perft: Kiwipete ===" << std::endl;
    for (int d = 1; d <= 4; d++) {
        setFEN(pos, kiwiFen);
        auto t0 = std::chrono::steady_clock::now();
        U64 got = perft(pos, d);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        bool pass = (got == kiwiExp[d - 1]);
        if (!pass) allPass = false;
        std::cout << "depth " << d << ": " << got << "  expected " << kiwiExp[d - 1]
                  << "  " << (pass ? "PASS" : "FAIL")
                  << "  (" << (long long)ms << " ms, " << (ms > 0 ? got / ms / 1000.0 : 0.0)
                  << " Mnps)\n";
    }

    std::cout << (allPass ? "ALL PERFT TESTS PASSED" : "SOME PERFT TESTS FAILED") << std::endl;
    return allPass;
}

// Additional standard perft positions (Chess Programming Wiki "Perft Results").
struct PerftCase {
    const char* fen;
    U64 expected[6];     // depth 1..6, 0 = stop
};

static const PerftCase EXTRA_CASES[] = {
    // Position 3: rook/pawn endgame, en passant heavy
    { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      { 14, 191, 2812, 43238, 674624, 11030083 } },
    // Position 4: promotions + castling
    { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      { 6, 264, 9467, 422333, 15833292, 706045033 } },
    // Position 5: castling + promotions + pins
    { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      { 44, 1486, 62379, 2103487, 89941194, 0 } },
    // Position 6: quiet middlegame
    { "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
      { 46, 2079, 89890, 3894594, 164075551, 0 } }
};
static const int EXTRA_MAX_DEPTH[] = {5, 4, 4, 4};

static bool runExtraPerftSuite() {
    bool allPass = true;
    Position pos;
    std::cout << "=== Perft: additional standard positions ===" << std::endl;
    for (size_t i = 0; i < sizeof(EXTRA_CASES) / sizeof(EXTRA_CASES[0]); i++) {
        std::cout << "position " << (i + 1) << ": " << EXTRA_CASES[i].fen << std::endl;
        for (int d = 1; d <= EXTRA_MAX_DEPTH[i]; d++) {
            setFEN(pos, EXTRA_CASES[i].fen);
            auto t0 = std::chrono::steady_clock::now();
            U64 got = perft(pos, d);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            U64 expect = EXTRA_CASES[i].expected[d - 1];
            bool pass = (got == expect);
            if (!pass) allPass = false;
            std::cout << "  depth " << d << ": " << got << "  expected " << expect
                      << "  " << (pass ? "PASS" : "FAIL")
                      << "  (" << (long long)ms << " ms, "
                      << (ms > 0 ? got / ms / 1000.0 : 0.0) << " Mnps)\n";
        }
    }
    std::cout << (allPass ? "ALL EXTRA PERFT TESTS PASSED"
                          : "SOME EXTRA PERFT TESTS FAILED") << std::endl;
    return allPass;
}

// ---------------------------------------------------------------------------
// Internal consistency validation
//   - incremental Zobrist hash vs. recomputation from scratch
//   - board[]/colorOf[] vs. piece bitboards vs. occupancy
//   - make()/unmake() round trips inside randomly played games
// ---------------------------------------------------------------------------
static bool validatePosition(const Position& pos) {
    if (pos.computeHash() != pos.hash) return false;
    U64 all = pos.occ[WHITE] | pos.occ[BLACK];
    if (all != pos.occ[2]) return false;
    if (pos.occ[WHITE] & pos.occ[BLACK]) return false;
    for (int s = 0; s < 64; s++) {
        int b = pos.board[s];
        if (b == EMPTY) {
            if (all & bit(s)) return false;
        } else {
            int c = pos.colorOf[s];
            if (c != WHITE && c != BLACK) return false;
            if (!(pos.pieces[c][b] & bit(s))) return false;
        }
    }
    for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++) {
        U64 bb = pos.pieces[c][p];
        while (bb) {
            int s = poplsb(bb);
            if (pos.board[s] != p || pos.colorOf[s] != c) return false;
        }
    }
    return true;
}

static bool samePosition(const Position& a, const Position& b) {
    bool ok = true;
    if (a.hash != b.hash)             { std::cout << "   hash " << a.hash << " != " << b.hash << "\n"; ok = false; }
    if (a.side != b.side)             { std::cout << "   side\n"; ok = false; }
    if (a.castling != b.castling)     { std::cout << "   castling " << a.castling << " != " << b.castling << "\n"; ok = false; }
    if (a.epSquare != b.epSquare)     { std::cout << "   epSquare " << a.epSquare << " != " << b.epSquare << "\n"; ok = false; }
    if (a.halfmove != b.halfmove)     { std::cout << "   halfmove " << a.halfmove << " != " << b.halfmove << "\n"; ok = false; }
    if (a.fullmove != b.fullmove)     { std::cout << "   fullmove " << a.fullmove << " != " << b.fullmove << "\n"; ok = false; }
    if (a.histPly != b.histPly)       { std::cout << "   histPly " << a.histPly << " != " << b.histPly << "\n"; ok = false; }
    if (a.histPly == b.histPly) {
        for (int i = 0; i < a.histPly; i++)
            if (a.keyHist[i] != b.keyHist[i]) { std::cout << "   keyHist[" << i << "]\n"; ok = false; break; }
    }
    for (int c = 0; c < 2; c++) for (int p = 0; p < 6; p++)
        if (a.pieces[c][p] != b.pieces[c][p]) { std::cout << "   pieces[" << c << "][" << p << "]\n"; ok = false; }
    for (int i = 0; i < 3; i++)
        if (a.occ[i] != b.occ[i]) { std::cout << "   occ[" << i << "]\n"; ok = false; }
    for (int s = 0; s < 64; s++) {
        if (a.board[s] != b.board[s])     { std::cout << "   board[" << s << "]\n"; ok = false; break; }
        if (a.colorOf[s] != b.colorOf[s]) { std::cout << "   colorOf[" << s << "]\n"; ok = false; break; }
    }
    return ok;
}

static bool consistencyCheck(int games, int maxPlies) {
    static const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3"
    };
    const int nFens = (int)(sizeof(fens) / sizeof(fens[0]));
    std::cout << "=== Consistency check (" << games << " random games) ===" << std::endl;
    for (int g = 0; g < games; g++) {
        Position pos;
        setFEN(pos, fens[g % nFens]);
        for (int ply = 0; ply < maxPlies; ply++) {
            if (!validatePosition(pos)) {
                std::cout << "FAIL: inconsistent position (game " << g
                          << ", ply " << ply << ")" << std::endl;
                return false;
            }
            MoveList legal;
            genLegal(pos, legal);
            if (legal.n == 0) break;                      // checkmate or stalemate
            Move m = legal.moves[rng() % (U64)legal.n];
            Position before = pos;
            Undo u;
            makeMove(pos, m, u);
            if (!validatePosition(pos)) {
                std::cout << "FAIL: inconsistent after make " << moveToStr(m) << std::endl;
                return false;
            }
            unmakeMove(pos, m, u);
            if (!samePosition(before, pos)) {
                std::cout << "FAIL: unmake did not restore position after "
                          << moveToStr(m) << std::endl;
                return false;
            }
            makeMove(pos, m, u);                          // play it and continue
        }
    }
    std::cout << "CONSISTENCY CHECK PASSED" << std::endl;
    return true;
}

static void benchPerft(const char* fen, int depth) {
    Position pos;
    setFEN(pos, fen);
    auto t0 = std::chrono::steady_clock::now();
    U64 nodes = perft(pos, depth);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "perft(" << depth << "): " << nodes << " nodes in " << secs
              << " s -> " << (nodes / secs) << " nodes/sec ("
              << (nodes / secs / 1e6) << " Mnps)" << std::endl;
}

// Aggregate, reproducible benchmark (like a "bench" in other engines).
static void runBench() {
    struct BenchCase { const char* fen; int depth; };
    static const BenchCase cases[] = {
        { START_FEN, 6 },
        { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5 },
        { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5 },
        { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4 },
        { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4 },
        { "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4 }
    };
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));
    Position pos;
    U64 total = 0;
    std::cout << "=== Bench ===" << std::endl;
    auto benchStart = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        setFEN(pos, cases[i].fen);
        auto t0 = std::chrono::steady_clock::now();
        U64 nodes = perft(pos, cases[i].depth);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total += nodes;
        std::cout << "perft(" << cases[i].depth << ") nodes " << nodes
                  << "  " << (long long)ms << " ms  "
                  << (ms > 0 ? nodes / ms / 1000.0 : 0.0) << " Mnps" << std::endl;
    }
    auto benchEnd = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(benchEnd - benchStart).count();
    std::cout << "Total: " << total << " nodes in " << secs << " s -> "
              << (U64)(total / secs) << " nodes/sec ("
              << (total / secs / 1e6) << " Mnps)" << std::endl;
}

// ---------------------------------------------------------------------------
// Evaluation: material + piece-square tables, tapered by game phase.
// Pluggable: replace eval() to experiment without touching search.
// Scores are centipawns from White's perspective (sign-flipped by side).
// ---------------------------------------------------------------------------
static const int MG_VAL[6] = {82, 337, 365, 477, 1025, 0};
static const int EG_VAL[6] = {94, 281, 297, 512,  936, 0};

// Tables written from White's perspective, indexed [square] with a1 = 0.
static const int MG_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0 };
static const int EG_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
   178,173,158,134,147,132,165,187,
    94,100, 85, 67, 56, 53, 82, 84,
    32, 24, 13,  5, -2,  4, 17, 17,
    13,  9, -3, -7, -7, -8,  3, -1,
     4,  7, -6,  1,  0, -5, -1, -8,
    13,  8,  8, 10, 13,  0,  2, -7,
     0,  0,  0,  0,  0,  0,  0,  0 };
static const int MG_KNIGHT[64] = {
   -167,-89,-34,-49, 61,-97,-15,-107,
    -73,-41, 72, 36, 23, 62,  7, -17,
    -47, 60, 37, 65, 84,129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22,
    -13,  4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25,-16,
    -29,-53,-12, -3, -1, 18,-14,-19,
   -105,-21,-58,-33,-17,-28,-19,-23 };
static const int EG_KNIGHT[64] = {
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20, 10,  9, -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64 };
static const int MG_BISHOP[64] = {
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21 };
static const int EG_BISHOP[64] = {
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17 };
static const int MG_ROOK[64] = {
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26 };
static const int EG_ROOK[64] = {
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  3,  8,  3,
      7,  7,  7,  5,  4, -3, -5, -3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8,-11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20 };
static const int MG_QUEEN[64] = {
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14,  2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50 };
static const int EG_QUEEN[64] = {
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41 };
static const int MG_KING[64] = {
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14 };
static const int EG_KING[64] = {
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43 };

static const int* MG_PST[6] = {MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING};
static const int* EG_PST[6] = {EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING};

static const int GAME_PHASE[6] = {0, 1, 1, 2, 4, 0};

static inline int mirrorSq(int s) { return s ^ 56; }

// --- learnable-evaluation hooks (implemented after the hand-tuned eval) -----
static bool useLearnedEval = false;                 // set via UCI option
static int  evaluateLearned(const Position& pos);   // forward declaration

static int evaluate(const Position& pos) {
    if (useLearnedEval) return evaluateLearned(pos);
    int mg[2] = {0, 0}, eg[2] = {0, 0}, phase = 0;
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < 6; p++) {
            U64 b = pos.pieces[c][p];
            while (b) {
                int s = poplsb(b);
                // PSTs are written rank-8-first, so a White piece's table index
                // is its square flipped vertically; a Black piece's is the
                // square's own index (it is already stored mirrored).
                int ts = (c == WHITE) ? (s ^ 56) : s;
                mg[c] += MG_VAL[p] + MG_PST[p][ts];
                eg[c] += EG_VAL[p] + EG_PST[p][ts];
                phase += GAME_PHASE[p];
            }
        }
    }
    if (phase > 24) phase = 24;
    int mgScore = mg[WHITE] - mg[BLACK];
    int egScore = eg[WHITE] - eg[BLACK];
    int score = (mgScore * phase + egScore * (24 - phase)) / 24;
    return (pos.side == WHITE) ? score : -score;   // relative to side to move
}

// ---------------------------------------------------------------------------
// Learnable evaluation
//
// The hand-tuned eval above is a tapered linear model:  score = sum of
// (mg/eg material) + (mg/eg piece-square) + tempo, interpolated by game phase.
// Exactly those parameters can be exported to a text file, trained in Python
// (see ml/) and loaded back at runtime, which turns this engine into the
// "referee + search" half of a machine-learning loop:
//
//   weights index layout (EVAL_NF = 1562 doubles, version 1):
//     [   0 ..  11] mg material   : [colour * 6 + pieceType]
//     [  12 ..  23] eg material   : [colour * 6 + pieceType]
//     [  24 .. 791] mg PST        : [colour][pieceType][square-in-own-frame]
//     [ 792 .. 1559] eg PST       : [colour][pieceType][square-in-own-frame]
//     [1560]        mg tempo bonus
//     [1561]        eg tempo bonus
//
// "square-in-own-frame" is the white-frame index: for White it is `sq ^ 56`
// (the tables are written rank-8 first) and for Black it is `sq` itself.
// ---------------------------------------------------------------------------
static const int EVAL_NF_MAT   = 2 * 12;
static const int EVAL_NF_PST   = 2 * 2 * 6 * 64;
static const int EVAL_NF_TEMPO = 2;
static const int EVAL_NF       = EVAL_NF_MAT + EVAL_NF_PST + EVAL_NF_TEMPO;   // 1562

static inline int W_MAT(int ph, int c, int p)         { return ph * 12 + c * 6 + p; }
static inline int W_PST(int ph, int c, int p, int s)  { return EVAL_NF_MAT + ph * 768 + c * 384 + p * 64 + s; }
static inline int W_TEMPO(int ph)                     { return EVAL_NF_MAT + EVAL_NF_PST + ph; }

static std::vector<double> learnedW;      // empty when no weight file is loaded
static std::string         evalFileName;

static int evaluateLearned(const Position& pos) {
    double mg[2] = {0.0, 0.0}, eg[2] = {0.0, 0.0};
    int phase = 0;
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < 6; p++) {
            U64 b = pos.pieces[c][p];
            int cnt = popcount(b);
            if (!cnt) continue;
            mg[c] += cnt * learnedW[W_MAT(0, c, p)];
            eg[c] += cnt * learnedW[W_MAT(1, c, p)];
            phase += cnt * GAME_PHASE[p];
            while (b) {
                int s = poplsb(b);
                int ts = (c == WHITE) ? (s ^ 56) : s;
                mg[c] += learnedW[W_PST(0, c, p, ts)];
                eg[c] += learnedW[W_PST(1, c, p, ts)];
            }
        }
    }
    if (phase > 24) phase = 24;
    // Interpolate in integer arithmetic (truncating division) so that, at
    // initialization, the learned model reproduces the hand-tuned eval exactly.
    long long mgW = (long long)llround(mg[WHITE] - mg[BLACK]) + (long long)llround(learnedW[W_TEMPO(0)]);
    long long egW = (long long)llround(eg[WHITE] - eg[BLACK]) + (long long)llround(learnedW[W_TEMPO(1)]);
    long long score = (mgW * phase + egW * (24 - phase)) / 24;
    if (pos.side == BLACK) score = -score;
    return (int)score;
}

// Build the weight vector that reproduces the hand-tuned evaluation exactly.
static std::vector<double> handTunedWeights() {
    std::vector<double> w(EVAL_NF, 0.0);
    for (int ph = 0; ph < 2; ph++)
        for (int c = 0; c < 2; c++)
            for (int p = 0; p < 6; p++)
                w[W_MAT(ph, c, p)] = (ph == 0) ? MG_VAL[p] : EG_VAL[p];
    for (int ph = 0; ph < 2; ph++)
        for (int c = 0; c < 2; c++)
            for (int p = 0; p < 6; p++)
                for (int s = 0; s < 64; s++)
                    w[W_PST(ph, c, p, s)] = (ph == 0) ? MG_PST[p][s] : EG_PST[p][s];
    return w;   // tempo stays 0
}

static bool loadEvalFile(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) {
        std::cout << "info string error: cannot open eval file " << path << std::endl;
        return false;
    }
    std::string line;
    std::vector<double> vals;
    while (std::getline(in, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::istringstream ls(line);
        double v;
        while (ls >> v) vals.push_back(v);
    }
    if ((int)vals.size() == EVAL_NF + 1) vals.erase(vals.begin());   // drop version
    if ((int)vals.size() != EVAL_NF) {
        std::cout << "info string error: eval file " << path << " has "
                  << vals.size() << " values, expected " << EVAL_NF << std::endl;
        return false;
    }
    learnedW = vals;
    evalFileName = path;
    std::cout << "info string loaded eval weights from " << path << std::endl;
    return true;
}

static bool saveEvalFile(const std::string& path) {
    std::ofstream out(path.c_str());
    if (!out) return false;
    std::vector<double> w = learnedW.empty() ? handTunedWeights() : learnedW;
    out << "# SBEVAL v1 n=" << EVAL_NF << "\n";
    out << std::fixed;
    out.precision(6);
    for (int i = 0; i < EVAL_NF; i++) {
        out << w[i];
        out << ((i % 6 == 5) ? '\n' : ' ');
    }
    return true;
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
enum { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    U64     key   = 0;
    Move    best  = MOVE_NONE;
    I16     score = 0;
    int8_t  depth = 0;
    uint8_t meta  = TT_EXACT;   // bits 0-1: flag, bits 2-7: generation age
};                              // 16 bytes exactly

static size_t               ttEntries = 0;   // power of two
static size_t               ttMask = 0;
static std::vector<TTEntry> TT;
static uint8_t              ttAge = 0;

static inline int  ttFlagOf(const TTEntry* e) { return e->meta & 3; }
static inline int  ttAgeOf(const TTEntry* e)  { return e->meta >> 2; }

static void ttResize(int megabytes) {
    if (megabytes < 1) megabytes = 1;
    if (megabytes > 1024) megabytes = 1024;
    size_t wanted = ((size_t)megabytes * 1024 * 1024) / sizeof(TTEntry);
    size_t p = 1;
    while (p * 2 <= wanted) p *= 2;          // round down to a power of two
    ttEntries = p;
    ttMask = p - 1;
    TT.assign(ttEntries, TTEntry());
    ttAge = 0;
}

static inline void ttClear() {
    if (ttEntries == 0) ttResize(32);
    TT.assign(ttEntries, TTEntry());
    ttAge = 0;
}

static inline TTEntry* ttProbe(U64 key) {
    TTEntry* e = &TT[key & ttMask];
    return (e->key == key) ? e : nullptr;
}

static inline void ttStore(U64 key, int depth, int score, int flag, Move best) {
    if (ttEntries == 0) return;
    TTEntry* e = &TT[key & ttMask];
    bool samePos = (e->key == key) && (ttAgeOf(e) == ttAge);
    if (!samePos || depth >= e->depth) {
        e->key = key;
        e->best = best;
        e->score = (I16)score;
        e->depth = (int8_t)(depth > 127 ? 127 : depth);
        e->meta = (uint8_t)(flag | (ttAge << 2));
    } else if (best != MOVE_NONE) {
        e->best = best;               // keep a better move hint for the same position
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
static const int INF = 32000;
static const int MATE = 30000;
static const int MAX_PLY = 128;

struct SearchState {
    std::chrono::steady_clock::time_point t0;
    double timeLimitMs = 1000;
    bool stop = false;
    bool timeManaged = false;
    U64 nodes = 0;
    Move killers[MAX_PLY][2];
    int history[2][64][64];
    Move pv[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
};

static void clearSearchState(SearchState& st) {
    memset(st.killers, 0, sizeof(st.killers));
    memset(st.history, 0, sizeof(st.history));
    memset(st.pv, 0, sizeof(st.pv));
    memset(st.pvLen, 0, sizeof(st.pvLen));
    st.nodes = 0;
    st.stop = false;
}

static double elapsedMs(const SearchState& st) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - st.t0).count();
}

// ---------------------------------------------------------------------------
// Asynchronous stdin reader.  A background thread feeds lines into a queue so
// the search can react to "stop"/"quit" while it is thinking.  This works
// identically for pipes (GUIs), redirected files and interactive consoles.
// ---------------------------------------------------------------------------
static std::deque<std::string> g_lines;
static bool g_eof = false;

#ifdef _WIN32
static CRITICAL_SECTION g_cs;
static void inputLock()   { EnterCriticalSection(&g_cs); }
static void inputUnlock() { LeaveCriticalSection(&g_cs); }
#else
static std::mutex g_mtx;
static void inputLock()   { g_mtx.lock(); }
static void inputUnlock() { g_mtx.unlock(); }
#endif

static void sleepMs(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    usleep(ms * 1000);
#endif
}

static void readerLoop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        inputLock();
        g_lines.push_back(line);
        inputUnlock();
    }
    inputLock();
    g_eof = true;
    inputUnlock();
}

#ifdef _WIN32
static DWORD WINAPI readerThreadProc(LPVOID) { readerLoop(); return 0; }
#endif

static void startInputReader() {
#ifdef _WIN32
    InitializeCriticalSection(&g_cs);
    CreateThread(NULL, 0, readerThreadProc, NULL, 0, NULL);
#else
    std::thread(readerLoop).detach();
#endif
}

enum LineStatus { LS_NONE, LS_LINE, LS_EOF };

// wait == true  : block until a line arrives or EOF is reached
// wait == false : return immediately if nothing is available
static LineStatus pollLine(std::string& out, bool wait) {
    for (;;) {
        inputLock();
        if (!g_lines.empty()) {
            out = g_lines.front();
            g_lines.pop_front();
            inputUnlock();
            return LS_LINE;
        }
        bool eof = g_eof;
        inputUnlock();
        if (eof) return LS_EOF;
        if (!wait) return LS_NONE;
        sleepMs(1);
    }
}

// Commands received while a search is running are buffered here and executed
// by the main loop once the search has finished (nothing is ever dropped).
static std::vector<std::string> cmdQueue;

static void handleStopCommand(const std::string& line, SearchState& st, bool& quitFlag) {
    std::istringstream ss(line);
    std::string tok;
    ss >> tok;
    if (tok == "stop") { st.stop = true; }
    else if (tok == "quit") { st.stop = true; quitFlag = true; }
    else if (!tok.empty()) { cmdQueue.push_back(line); }
}

static bool checkStop(SearchState& st, bool& quitFlag) {
    if (st.stop) return true;
    if ((st.nodes & 2047) == 0) {
        if (st.timeManaged && elapsedMs(st) >= st.timeLimitMs) { st.stop = true; return true; }
        std::string line;
        if (pollLine(line, false) == LS_LINE) handleStopCommand(line, st, quitFlag);
    }
    return st.stop;
}

// Mate score adjustment for TT storage/retrieval
static inline int scoreToTT(int score, int ply) {
    if (score > MATE - MAX_PLY) return score + ply;
    if (score < -MATE + MAX_PLY) return score - ply;
    return score;
}
static inline int scoreFromTT(int score, int ply) {
    if (score > MATE - MAX_PLY) return score - ply;
    if (score < -MATE + MAX_PLY) return score + ply;
    return score;
}

static const int SEE_VAL[6] = {100, 320, 330, 500, 900, 0};

static void scoreMoves(const Position& pos, MoveList& ml, Move ttMove, int ply,
                       const SearchState& st) {
    for (int i = 0; i < ml.n; i++) {
        Move m = ml.moves[i];
        int score = 0;
        if (m == ttMove) score = 10000000;
        else if (mvIsCap(m)) {
            int victim = (mvFlags(m) == 2) ? PAWN : pos.board[mvTo(m)];
            int attacker = pos.board[mvFrom(m)];
            score = 1000000 + SEE_VAL[victim] * 10 - SEE_VAL[attacker] / 10;
            if (mvPromo(m)) score += SEE_VAL[mvPromo(m)];
        }
        else if (mvPromo(m)) score = 900000 + SEE_VAL[mvPromo(m)];
        else if (m == st.killers[ply][0]) score = 800000;
        else if (m == st.killers[ply][1]) score = 700000;
        else score = st.history[pos.side][mvFrom(m)][mvTo(m)];
        ml.scores[i] = score;
    }
}

static inline Move pickMove(MoveList& ml, int idx) {
    int best = idx;
    for (int i = idx + 1; i < ml.n; i++)
        if (ml.scores[i] > ml.scores[best]) best = i;
    Move m = ml.moves[best];
    int s = ml.scores[best];
    ml.moves[best] = ml.moves[idx];
    ml.scores[best] = ml.scores[idx];
    ml.moves[idx] = m;
    ml.scores[idx] = s;
    return m;
}

// ---------------------------------------------------------------------------
// Quiescence search (captures only, with stand-pat)
// ---------------------------------------------------------------------------
static int qsearch(Position& pos, SearchState& st, int alpha, int beta, int ply,
                   bool& quitFlag) {
    st.nodes++;
    if (checkStop(st, quitFlag)) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    int standPat = evaluate(pos);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;

    MoveList ml;
    genPseudo(pos, ml, true);   // captures only
    MoveList legal;
    int us = pos.side;
    legal.n = 0;
    for (int i = 0; i < ml.n; i++) {
        Undo u;
        makeMove(pos, ml.moves[i], u);
        if (!pos.inCheck(us)) legal.add(ml.moves[i]);
        unmakeMove(pos, ml.moves[i], u);
    }
    scoreMoves(pos, legal, MOVE_NONE, ply, st);

    for (int i = 0; i < legal.n; i++) {
        Move m = pickMove(legal, i);
        Undo u;
        makeMove(pos, m, u);
        int score = -qsearch(pos, st, -beta, -alpha, ply + 1, quitFlag);
        unmakeMove(pos, m, u);
        if (st.stop) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ---------------------------------------------------------------------------
// Alpha-beta (negamax, fail-soft, PVS-free for clarity)
// ---------------------------------------------------------------------------
static int alphabeta(Position& pos, SearchState& st, int depth, int alpha, int beta,
                     int ply, bool& quitFlag) {
    if (checkStop(st, quitFlag)) return 0;
    st.pvLen[ply] = ply;

    if (ply >= MAX_PLY - 1) return evaluate(pos);
    if (depth <= 0) return qsearch(pos, st, alpha, beta, ply, quitFlag);
    if (pos.halfmove >= 100) return 0;            // fifty-move rule
    if (ply > 0 && isRepetition(pos)) return 0;   // repetition -> draw

    bool inChk = pos.inCheck(pos.side);

    // TT probe
    Move ttMove = MOVE_NONE;
    TTEntry* tte = ttProbe(pos.hash);
    if (tte) {
        ttMove = tte->best;
        if (tte->depth >= depth && ply > 0) {
            int tscore = scoreFromTT(tte->score, ply);
            int tflag = ttFlagOf(tte);
            if (tflag == TT_EXACT) return tscore;
            if (tflag == TT_LOWER && tscore >= beta) return tscore;
            if (tflag == TT_UPPER && tscore <= alpha) return tscore;
        }
    }

    MoveList pseudo;
    genPseudo(pos, pseudo, false);
    int us = pos.side;

    // Legalize + score
    MoveList legal;
    legal.n = 0;
    for (int i = 0; i < pseudo.n; i++) {
        Undo u;
        makeMove(pos, pseudo.moves[i], u);
        if (!pos.inCheck(us)) legal.add(pseudo.moves[i]);
        unmakeMove(pos, pseudo.moves[i], u);
    }
    if (legal.n == 0) return inChk ? (-MATE + ply) : 0;

    scoreMoves(pos, legal, ttMove, ply, st);

    int bestScore = -INF;
    Move bestMove = MOVE_NONE;
    int flag = TT_UPPER;
    int alphaOrig = alpha;
    (void)alphaOrig;

    for (int i = 0; i < legal.n; i++) {
        Move m = pickMove(legal, i);
        Undo u;
        makeMove(pos, m, u);
        st.nodes++;
        int score = -alphabeta(pos, st, depth - 1, -beta, -alpha, ply + 1, quitFlag);
        unmakeMove(pos, m, u);
        if (st.stop) return 0;

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
            // update PV
            st.pv[ply][ply] = m;
            for (int j = ply + 1; j < st.pvLen[ply + 1]; j++)
                st.pv[ply][j] = st.pv[ply + 1][j];
            st.pvLen[ply] = st.pvLen[ply + 1];
        }
        if (score > alpha) {
            alpha = score;
            flag = TT_EXACT;
        }
        if (alpha >= beta) {
            flag = TT_LOWER;
            // killers + history for quiet moves
            if (!mvIsCap(m) && !mvPromo(m)) {
                if (st.killers[ply][0] != m) {
                    st.killers[ply][1] = st.killers[ply][0];
                    st.killers[ply][0] = m;
                }
                st.history[us][mvFrom(m)][mvTo(m)] += depth * depth;
                if (st.history[us][mvFrom(m)][mvTo(m)] > 500000) {
                    // age history to avoid overflow
                    for (int c = 0; c < 2; c++)
                        for (int a = 0; a < 64; a++)
                            for (int b = 0; b < 64; b++) st.history[c][a][b] /= 2;
                }
            }
            break;
        }
    }

    ttStore(pos.hash, depth, scoreToTT(bestScore, ply), flag, bestMove);
    return bestScore;
}

// ---------------------------------------------------------------------------
// Iterative deepening driver
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Iterative deepening driver (single PV or MultiPV)
// ---------------------------------------------------------------------------
struct RootMove {
    Move move = MOVE_NONE;
    int  score = -INF;
    int  pvLen = 0;
    Move pv[MAX_PLY];
};

static int multiPV = 1;      // UCI option "MultiPV"

static void printInfoLine(const SearchState& st, int depth, int pvIndex, int score,
                          const Move* pv, int pvLen, double ms) {
    std::ostringstream info;
    info << "info depth " << depth;
    if (multiPV > 1) info << " multipv " << pvIndex;
    info << " score ";
    if (score > MATE - MAX_PLY) info << "mate " << (MATE - score + 1) / 2;
    else if (score < -MATE + MAX_PLY) info << "mate " << -(MATE + score + 1) / 2;
    else info << "cp " << score;
    info << " nodes " << st.nodes
         << " nps " << (unsigned long long)(st.nodes * 1000.0 / ms)
         << " time " << (unsigned long long)ms << " pv";
    for (int i = 0; i < pvLen; i++) info << " " << moveToStr(pv[i]);
    std::cout << info.str() << std::endl;      // std::endl flushes
}

static Move searchBestMove(Position& pos, int maxDepth, double timeMs, bool timeManaged,
                           bool& quitFlag) {
    SearchState st;
    clearSearchState(st);
    st.t0 = std::chrono::steady_clock::now();
    st.timeLimitMs = timeMs;
    st.timeManaged = timeManaged;
    ttAge++;

    MoveList legal;
    genLegal(pos, legal);
    if (legal.n == 0) return MOVE_NONE;               // checkmate or stalemate

    std::vector<RootMove> roots((size_t)legal.n);
    for (int i = 0; i < legal.n; i++) {
        roots[(size_t)i].move = legal.moves[i];
        roots[(size_t)i].score = -INF;
        roots[(size_t)i].pvLen = 0;
    }

    Move bestMove = roots[0].move;
    int  bestScore = 0;

    for (int depth = 1; depth <= maxDepth && !st.stop; depth++) {
        bool completed = true;

        for (size_t i = 0; i < roots.size(); i++) {
            Move m = roots[i].move;
            Undo u;
            makeMove(pos, m, u);
            st.nodes++;
            int score;
            if (multiPV > 1 || i == 0) {
                // exact score (full window)
                score = -alphabeta(pos, st, depth - 1, -INF, INF, 1, quitFlag);
            } else {
                // later moves only need to prove they do not beat the current best
                score = -alphabeta(pos, st, depth - 1, -INF, -bestScore, 1, quitFlag);
            }
            unmakeMove(pos, m, u);
            if (st.stop) { completed = false; break; }

            roots[i].score = score;
            roots[i].pv[0] = m;
            roots[i].pvLen = 1;
            int childLen = st.pvLen[1];
            for (int j = 1; j < childLen && roots[i].pvLen < MAX_PLY; j++)
                roots[i].pv[roots[i].pvLen++] = st.pv[1][j];
        }
        if (!completed) break;                        // discard partial iteration

        // Order: best score first; stable so ties keep the previous iteration's order.
        std::stable_sort(roots.begin(), roots.end(),
                         [](const RootMove& a, const RootMove& b) { return a.score > b.score; });
        bestMove = roots[0].move;
        bestScore = roots[0].score;
        ttStore(pos.hash, depth, scoreToTT(bestScore, 0), TT_EXACT, bestMove);

        double ms = elapsedMs(st);
        if (ms < 1) ms = 1;
        int shown = (multiPV > 1) ? (int)std::min<size_t>((size_t)multiPV, roots.size()) : 1;
        for (int k = 0; k < shown; k++)
            printInfoLine(st, depth, k + 1, roots[(size_t)k].score,
                          roots[(size_t)k].pv, roots[(size_t)k].pvLen, ms);

        if (bestScore > MATE - MAX_PLY) break;        // mate found, no need to go deeper
    }
    return bestMove;
}

static int ttSizeMB = 32;

// ---------------------------------------------------------------------------
// UCI protocol
// ---------------------------------------------------------------------------
static void uciSetOption(const std::string& args) {
    // Syntax: setoption name <id> [value <x>]
    std::istringstream ss(args);
    std::string tok, name, value;
    ss >> tok;                        // "name"
    if (tok != "name") return;
    while (ss >> tok && tok != "value") {
        if (!name.empty()) name += " ";
        name += tok;
    }
    while (ss >> tok) {
        if (!value.empty()) value += " ";
        value += tok;
    }
    if (name == "Hash") {
        int mb = atoi(value.c_str());
        if (mb > 0) { ttSizeMB = mb; ttResize(mb); }
    } else if (name == "Clear Hash") {
        ttClear();
    } else if (name == "MultiPV") {
        int n = atoi(value.c_str());
        multiPV = (n < 1) ? 1 : (n > 64 ? 64 : n);
    } else if (name == "EvalFile") {
        if (!value.empty() && loadEvalFile(value)) useLearnedEval = true;
    } else if (name == "UseLearnedEval") {
        bool want = (value == "true" || value == "1" || value == "True");
        if (want && learnedW.empty()) {
            std::cout << "info string error: no learned weights loaded (set EvalFile first)"
                      << std::endl;
        } else {
            useLearnedEval = want;
        }
    }
}
static Move uciStrToMove(Position& pos, const std::string& s) {
    MoveList ml;
    genLegal(pos, ml);
    for (int i = 0; i < ml.n; i++)
        if (moveToStr(ml.moves[i]) == s) return ml.moves[i];
    return MOVE_NONE;
}

static void uciGo(Position& pos, const std::string& args, bool& quitFlag) {
    std::istringstream ss(args);
    std::string tok;
    int depth = 64;
    double movetime = -1, wtime = -1, btime = -1, winc = 0, binc = 0;
    bool infinite = false;
    int movestogo = 30;
    while (ss >> tok) {
        if (tok == "depth") ss >> depth;
        else if (tok == "movetime") ss >> movetime;
        else if (tok == "wtime") ss >> wtime;
        else if (tok == "btime") ss >> btime;
        else if (tok == "winc") ss >> winc;
        else if (tok == "binc") ss >> binc;
        else if (tok == "movestogo") ss >> movestogo;
        else if (tok == "infinite") infinite = true;
    }

    bool timeManaged = true;
    double timeMs;
    if (movetime >= 0) {
        timeMs = movetime * 0.95;
    } else if (infinite) {
        timeManaged = false; timeMs = 0;
    } else {
        double myTime = (pos.side == WHITE) ? wtime : btime;
        double myInc  = (pos.side == WHITE) ? winc : binc;
        if (myTime < 0) { timeManaged = false; timeMs = 0; }
        else {
            timeMs = myTime / (double)movestogo + myInc * 0.8;
            if (timeMs > myTime * 0.5) timeMs = myTime * 0.5;
            if (timeMs < 10) timeMs = 10;
        }
    }

    Move best = searchBestMove(pos, depth, timeMs, timeManaged, quitFlag);
    std::cout << "bestmove " << moveToStr(best) << std::endl;
}

static void uciPosition(Position& pos, const std::string& args) {
    std::istringstream ss(args);
    std::string tok;
    ss >> tok;
    if (tok == "startpos") {
        setFEN(pos, START_FEN);
        tok.clear();
        ss >> tok;   // maybe "moves"
    } else if (tok == "fen") {
        std::string fen, word;
        int parts = 0;
        while (ss >> word && word != "moves" && parts < 6) {
            if (!fen.empty()) fen += " ";
            fen += word;
            parts++;
        }
        setFEN(pos, fen);
        if (word != "moves") return;
        tok = word;
    }
    if (tok == "moves") {
        std::string mv;
        while (ss >> mv) {
            Move m = uciStrToMove(pos, mv);
            if (m == MOVE_NONE) break;
            Undo u;
            makeMove(pos, m, u);
        }
    }
}

static void uciLoop() {
    Position pos;
    setFEN(pos, START_FEN);
    bool quitFlag = false;
    std::string line;

    while (!quitFlag) {
        if (!cmdQueue.empty()) {
            line = cmdQueue.front();
            cmdQueue.erase(cmdQueue.begin());
        } else {
            LineStatus ls = pollLine(line, true);
            if (ls != LS_LINE) break;   // EOF
        }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty()) continue;

        if (cmd == "uci") {
            std::cout << "id name SimpleBitboard 1.0\n"
                         "id author Cline\n"
                         "option name Hash type spin default 32 min 1 max 1024\n"
                         "option name Clear Hash type button\n"
                         "option name MultiPV type spin default 1 min 1 max 64\n"
                         "option name EvalFile type string default <empty>\n"
                         "option name UseLearnedEval type check default false\n"
                         "uciok" << std::endl;
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "setoption") {
            std::string rest;
            std::getline(ss, rest);
            uciSetOption(rest);
        } else if (cmd == "ucinewgame") {
            ttClear();
            setFEN(pos, START_FEN);
        } else if (cmd == "position") {
            std::string rest;
            std::getline(ss, rest);
            uciPosition(pos, rest);
        } else if (cmd == "go") {
            std::string rest;
            std::getline(ss, rest);
            uciGo(pos, rest, quitFlag);
        } else if (cmd == "stop") {
            // handled inside search; nothing to do here
        } else if (cmd == "quit") {
            quitFlag = true;
        } else if (cmd == "d") {
            printBoard(pos);
        } else if (cmd == "perft" || cmd == "divide") {
            int d = 5;
            ss >> d;
            perftDivide(pos, d);
        } else if (cmd == "eval") {
            std::cout << "eval: " << evaluate(pos) << " cp" << std::endl;
        } else if (cmd == "check") {
            consistencyCheck(20, 200);
        } else if (cmd == "hash") {
            std::cout << "incremental " << pos.hash << "  recomputed "
                      << pos.computeHash() << "  "
                      << (pos.hash == pos.computeHash() ? "OK" : "MISMATCH") << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    initAttacks();
    initZobrist();
    initCastlingClear();
    ttResize(ttSizeMB);

    if (argc > 1) {
        std::string a1 = argv[1];
        if (a1 == "test") {
            bool ok = runPerftSuite();
            ok = runExtraPerftSuite() && ok;
            return (ok && consistencyCheck(20, 240)) ? 0 : 1;
        }
        if (a1 == "check") {
            int games = (argc > 2) ? atoi(argv[2]) : 50;
            return consistencyCheck(games, 300) ? 0 : 1;
        }
        if (a1 == "perft") {
            int d = (argc > 2) ? atoi(argv[2]) : 5;
            const char* fen = (argc > 3) ? argv[3] : START_FEN;
            benchPerft(fen, d);
            return 0;
        }
        if (a1 == "bench") {
            runBench();
            return 0;
        }
        if (a1 == "dumpweights") {
            const char* path = (argc > 2) ? argv[2] : "weights_init.txt";
            if (saveEvalFile(path)) {
                std::cout << "wrote " << EVAL_NF << " hand-tuned weights to " << path
                          << std::endl;
                return 0;
            }
            std::cerr << "cannot write " << path << std::endl;
            return 1;
        }
    }

    startInputReader();
    uciLoop();
    return 0;
}

