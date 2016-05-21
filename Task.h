#ifndef TASK_H
#define TASK_H

#include <stdio.h>

class Task {
public:
  int thread_idx;
  float *buffer;
  size_t size;
  size_t offset;
  
  Task();
  Task(int thread_idx, float *buffer, size_t size, size_t offset);
  Task(const Task& task);
  virtual ~Task();
private:

};

#endif /* TASK_H */
