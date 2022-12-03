SOURCES=$(wildcard *.cpp)
OBJECTS=$(patsubst %.cpp, %.o, $(SOURCES))

BINARY=aurora.a

CPPFLAGS+=-std=c++20 -c -fpermissive -I. -I"../../affix-base/" -I"../../affix-base/asio/asio/include/"

all: $(BINARY)

$(BINARY): $(OBJECTS)
	# Builds the final library (.a file)
	ar rvs $(BINARY) $(OBJECTS)
	
%.o: %.cpp
	# Build a single .o file
	g++ $(CPPFLAGS) -o $@ $^
	
clean:
	rm -rf $(OBJECTS)
	rm -rf $(BINARY)

