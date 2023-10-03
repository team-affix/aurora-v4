B = main

S = main.cpp

H = ../aurora-v4/aurora.h
H += ../aurora-v4/fundamentals.h
H += ../aurora-v4/latent.h

F = -g -O2 -std=c++20 -I"../"

all: $B

$B: $S $H
	g++ $F $S -o $B

clean:
	rm -rf $B
