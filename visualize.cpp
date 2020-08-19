#include <GLUT/glut.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

//Needed for mouse rotation
bool fullscreen = false;
bool mouseDown = false;
float xrot = 0.0f;
float yrot = 0.0f;
float xdiff = 0.0f;
float ydiff = 0.0f;


struct Point {
    double x, y, z;
};


void read_points_from_txt(std::string &points_file, std::vector<Point> &points){
    std::ifstream in_file(points_file);
    if(!in_file.is_open()){
        throw std::runtime_error("Failed to open txt file with points.");
    }
    Point one_point{};
    while (in_file >> one_point.x && in_file >> one_point.y && in_file >> one_point.z){
        points.push_back(one_point);
    }
}


void drawPoints()
{
    std::vector<Point> points;
    std::string file_name = "../point_cloud.txt";
    read_points_from_txt(file_name, points);
    glBegin(GL_POINTS);
    for (auto & point : points){
        glColor3f(1, 1, 1);
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}


bool init(){
    glClearColor(0, 0, 0, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0f);
    return true;
}


void display(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    //змінити eyeZ шо вигляд поміняти - 1 це мало
//    gluLookAt (150, 150, 150, 0, 0, 0, 0, 100, 0); Устанавливает точку наблюдения, камеру.
//    Первые параметры откуда (x,y,z) и куда (x,y,z). Это пока главное.
    gluLookAt(
            0.0f, 0.0f, 1.5f,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.5f, 0.0f);

    glRotatef(xrot, 1.0f, 0.0f, 0.0f);
    glRotatef(yrot, 0.0f, 1.0f, 0.0f);
    drawPoints();
    glFlush();
    glutSwapBuffers();
}


void resize(int w, int h){
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    //gluPerspective (130, 1, 50, 0); Настройка перспективы просмотра. Нам сейчас интересны первые два параметра.
    // Первый параметр это охват в градусах от 0 до 180. Можете воспринимать его как обьектив на фотоаппарате.
    // Либо все но мелкое , либо большое но одно. Создав и запустив проект, поменяейте это параметр, вы увидите
    // изменение изображения больше-меньше. Посмотрите на рисунок ниже. Второй параметр это угол поворота по оси Y.
    // Да бог с ним. Главное первый параметр.
    gluPerspective(40.0f, 1.0f * w / h, 1.0f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void keyboard(unsigned char key, int x, int y)
{
    switch(key){
        case 27 :
            exit(1);
    }
}

void specialKeyboard(int key, int x, int y)
{
    if (key == GLUT_KEY_F1)
    {
        fullscreen = !fullscreen;
        if (fullscreen)
            glutFullScreen();
        else
        {
            glutReshapeWindow(500, 500);
            glutPositionWindow(50, 50);
        }
    }
}

void mouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        mouseDown = true;
        xdiff = x - yrot;
        ydiff = -y + xrot;
    }
    else
        mouseDown = false;
}

void mouseMotion(int x, int y)
{
    if (mouseDown)
    {
        yrot = x - xdiff;
        xrot = y + ydiff;
        glutPostRedisplay();
    }
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(500, 500);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutCreateWindow("Point cloud visualization");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
    glutReshapeFunc(resize);
    if (!init()){
        return -1;}
    glutMainLoop();
    return 0;
}

