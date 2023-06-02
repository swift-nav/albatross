#include <albatross/Common>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  albatross::blosc::init();
  ::testing::InitGoogleTest(&argc, argv);
  const int ret = RUN_ALL_TESTS();
  albatross::blosc::cleanup();
  return ret;
};