#include "Task.h"

Task::Task() {
  this->buffer = nullptr;
  this->size = 0;
  this->offset = 0;
}

Task::Task(int thread_idx, float *buffer, size_t size, size_t offset) {
  this->thread_idx = thread_idx;
  this->buffer = buffer;
  this->size = size;
  this->offset = offset;
}

Task::Task(const Task& task) {
  this->buffer = task.buffer;
  this->size = task.size;
  this->offset = task.offset;
}

Task::~Task() {
}
