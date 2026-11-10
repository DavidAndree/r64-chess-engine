#include <iostream>
#include <GLUT/glut.h>
#include <OpenGL/glu.h>

using namespace std;

//global variables for window width and height
int width = 640;
int height = 480;

//global variable for the chess board
int board[8][8];

//global variables for the chess board position and size
double board_x = 0.5 * (width - height);
double board_y = 0;
double board_size = height;

double square_size = board_size * 0.9 * 0.125; //size of each square on the chess board
double square_x = board_x + board_size * 0.05;
double square_y = board_y + board_size * 0.05;

void draw()
{
    //clear the window
    glClear(GL_COLOR_BUFFER_BIT);

    //draw a chess board
    glColor3f(1.0, 1.0, 1.0); //white color
    glBegin(GL_QUADS);
    glVertex2f(board_x, board_y);
    glVertex2f(board_x + board_size, board_y);
    glVertex2f(board_x + board_size, board_y + board_size);
    glVertex2f(board_x, board_y + board_size);
    glEnd();

    glBegin(GL_QUADS);
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
        {
            if ((x + y) % 2) glColor3f(0.05, 0.3, 0.05);
            else glColor3f(0.7, 0.65, 0.55);

            glVertex2f(square_x + square_size * x, square_y + square_size * y);
            glVertex2f(square_x + square_size * (x + 1), square_y + square_size * y);
            glVertex2f(square_x + square_size * (x + 1), square_y + square_size * (y + 1));
            glVertex2f(square_x + square_size * x, square_y + square_size * (y + 1));
        }

    
    //update the window
    glutSwapBuffers();

}
//display callback function
void display()
{
    draw();
}

void mouse(int button, int state, int x, int y)
{
    //handle mouse input
    if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
        cout << "Mouse clicked at (" << x << ", " << y << ")" << endl;
    }
}

void keydown(unsigned char key, int x, int y)
{
    //handle keyboard input
    if (key == 27) exit(0); //exit on ESC key
}

void timer(int value)
{
    //redraw the window
    glutPostRedisplay();

    //call the timer function again after 16 milliseconds
    glutTimerFunc(16, timer, 0);
}

int main(int argc, char** argv)
{
    //initialize OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

    //create the window
    glutInitWindowSize(width, height);
    glutCreateWindow("Chess Engine");

    glClearColor(0.0, 0.0, 0.0, 1.0);

    gluOrtho2D (0.0, width, 0.0, height);

    //set the display callback
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keydown);
    glutTimerFunc(16, timer, 0);

    //enter the main loop
    glutMainLoop();

    //end program
    return 0;
}