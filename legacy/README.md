# Legacy: OpenGL/GLUT board prototype

The first prototype of this repo: an OpenGL window that draws an empty chess board.
It is not part of r64 anymore and is kept only for reference.

Build and run (macOS):

```bash
clang++ -std=c++17 glut_board.cpp -o glut_board -framework GLUT -framework OpenGL -Wno-deprecated-declarations
./glut_board
```
