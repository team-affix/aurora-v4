SOURCES=$(wildcard *.cpp)
OBJECTS=$(patsubst %.cpp, %.o, $(SOURCES))
CRYPTOPP_LIB=../../affix-base/cryptopp/libcryptopp.a
AFFIX_BASE_LIB=../../affix-base/affix-base/affix-base.a
AURORA_LIB=../aurora-v4/aurora.a

CPPFLAGS+=-g
CPPFLAGS+=-O2
CPPFLAGS+=-std=c++20
CPPFLAGS+=-pthread
CPPFLAGS+=-I.
CPPFLAGS+=-I"../"
CPPFLAGS+=-I"../../affix-base/"
CPPFLAGS+=-I"../../affix-base/asio/asio/include/"

BINARY=main

all: $(BINARY)

$(BINARY): $(OBJECTS) $(AFFIX_BASE_LIB) $(CRYPTOPP_LIB) $(AURORA_LIB)
	# Build the final binary
	g++ $(CPPFLAGS) -o $@ $^
	
%.o: %.cpp
	# Build a single object file
	g++ $(CPPFLAGS) -c -o $@ $^
	
$(CRYPTOPP_LIB):
	# Build cryptopp
	make -C ../cryptopp/
	
$(AFFIX_BASE_LIB):
	# Build affix-base
	make -C ../../affix-base/
	
$(AURORA_LIB):
	# Build aurora-v4
	make -C ../aurora-v4/
	
clean:
	rm -rf *.o
	rm -rf $(BINARY)
