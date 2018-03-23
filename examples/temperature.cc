#include <fstream>
#include <sstream>
#include <iostream>
#include "csv.h"
#include "gflags/gflags.h"

DEFINE_string(input, "", "path to csv containing slant iono observations");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "reading data from : " << FLAGS_input << std::endl;

  return 0;
}
