#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
typedef Eigen::Matrix<GLfloat, 2, 1, Eigen::ColMajor> GLVector2f;
typedef Eigen::Matrix<GLfloat, 3, 1, Eigen::ColMajor> GLVector3f;
typedef Eigen::Matrix<GLfloat, 4, 1, Eigen::ColMajor> GLVector4f;
typedef Eigen::Matrix<GLfloat, 2, 2, Eigen::ColMajor> GLMatrix2f;
typedef Eigen::Matrix<GLfloat, 3, 3, Eigen::ColMajor> GLMatrix3f;
typedef Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> GLMatrix4f;

class Shader {
public:
  unsigned int program_;

  Shader() {}
  using Ptr = std::shared_ptr<Shader>;
  using ConstPtr = std::shared_ptr<const Shader>;
  // ------------------------------------------------------------------------
  // generate the shader on the fly
  void loadShaders(const char *vertex_file_path,
                   const char *fragment_file_path) {

    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);

    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from the file
    std::string VertexShaderCode;
    std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
    if (VertexShaderStream.is_open()) {
      std::string Line = "";
      while (getline(VertexShaderStream, Line))
        VertexShaderCode += "\n" + Line;
      VertexShaderStream.close();
    } else {
      printf("Impossible to open %s. Are you in the right directory ? Don't "
             "forget to read the FAQ !\n",
             vertex_file_path);
      getchar();
      return;
    }

    // Read the Fragment Shader code from the file
    std::string FragmentShaderCode;
    std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
    if (FragmentShaderStream.is_open()) {
      std::string Line = "";
      while (getline(FragmentShaderStream, Line))
        FragmentShaderCode += "\n" + Line;
      FragmentShaderStream.close();
    }

    GLint Result = GL_FALSE;
    int InfoLogLength;
    // Compile Vertex Shader
    printf("Compiling shader : %s\n", vertex_file_path);
    char const *VertexSourcePointer = VertexShaderCode.c_str();
    glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
    glCompileShader(VertexShaderID);

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
      std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
      glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL,
                         &VertexShaderErrorMessage[0]);
      printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    // Compile Fragment Shader
    printf("Compiling shader : %s\n", fragment_file_path);
    char const *FragmentSourcePointer = FragmentShaderCode.c_str();
    glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
      std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
      glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL,
                         &FragmentShaderErrorMessage[0]);
      printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    printf("Linking program ... \n");
    program_ = glCreateProgram();
    glAttachShader(program_, VertexShaderID);
    glAttachShader(program_, FragmentShaderID);
    glLinkProgram(program_);

    // Check the program
    glGetProgramiv(program_, GL_LINK_STATUS, &Result);
    glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 0) {
      std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
      glGetProgramInfoLog(program_, InfoLogLength, NULL,
                          &ProgramErrorMessage[0]);
      printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDetachShader(program_, VertexShaderID);
    glDetachShader(program_, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);
  }
  // ------------------------------------------------------------------------
  // activate the shader
  void useProgram() const { glUseProgram(program_); }
  void deleteProgram() { glDeleteProgram(program_); }
  // ------------------------------------------------------------------------
  // utility uniform functions
  // ------------------------------------------------------------------------
  void setBool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(program_, name.c_str()), (int)value);
  }
  // ------------------------------------------------------------------------
  void setInt(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(program_, name.c_str()), value);
  }
  // ------------------------------------------------------------------------
  void setFloat(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(program_, name.c_str()), value);
  }
  // ------------------------------------------------------------------------
  void setVec2(const std::string &name, const GLVector2f &value) const {
    glUniform2fv(glGetUniformLocation(program_, name.c_str()), 1, value.data());
  }
  void setVec2(const std::string &name, float x, float y) const {
    glUniform2f(glGetUniformLocation(program_, name.c_str()), x, y);
  }
  // ------------------------------------------------------------------------
  void setVec3(const std::string &name, const GLVector3f &value) const {
    glUniform3fv(glGetUniformLocation(program_, name.c_str()), 1, value.data());
  }
  void setVec3(const std::string &name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(program_, name.c_str()), x, y, z);
  }
  // ------------------------------------------------------------------------
  void setVec4(const std::string &name, const GLVector4f &value) const {
    glUniform4fv(glGetUniformLocation(program_, name.c_str()), 1, value.data());
  }
  void setVec4(const std::string &name, float x, float y, float z,
               float w) const {
    glUniform4f(glGetUniformLocation(program_, name.c_str()), x, y, z, w);
  }
  // ------------------------------------------------------------------------
  void setMat2(const std::string &name, const GLMatrix2f &mat) const {
    glUniformMatrix2fv(glGetUniformLocation(program_, name.c_str()), 1,
                       GL_FALSE, mat.data());
  }
  // ------------------------------------------------------------------------

  void setMat3(const std::string &name, const GLMatrix3f &mat) const {
    glUniformMatrix3fv(glGetUniformLocation(program_, name.c_str()), 1,
                       GL_FALSE, mat.data());
  }
  // ------------------------------------------------------------------------

  void setMat4(const std::string &name, const GLMatrix4f &mat) const {
    glUniformMatrix4fv(glGetUniformLocation(program_, name.c_str()), 1,
                       GL_FALSE, mat.data());
  }
};

#endif
