# ColorCorrect (CLI version)
 This program mathematically rebalances the color spectra in a picture via a per-channel percentile methods (default) or the gray-world method (with the -g flag)

 To compile, make sure you have a working OpenCV setup. If not, this can be installed via homebrew (on MacOS):

 ```
 brew install opencv pkg-config
 ```

 To compile, use the following command:

 ```
 g++ -std=c++11 colorcorrect.cpp -o colorcorrect `pkg-config --cflags --libs opencv4`
 ```
