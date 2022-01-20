# Build the include files
echo "Building the include files..."
deggl -i ../Eule/*.cpp -o Eule

# Verify that they compile cleanly
echo "Verifying that they compile"
g++ Eule.cpp -c -S -o - -Wall -Wextra -Wpedantic -mavx > /dev/null

