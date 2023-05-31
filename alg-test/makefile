B = main

S = main.cpp

H = ../aurora-v4/aurora.h
H += ../aurora-v4/fundamentals.h
H += ../aurora-v4/latent.h
H += ../aurora-v4/oneshot.h

F = -g -std=c++20 -I"../"

all: $B

$B: $S $H
	g++ $F $S -o $B

clean:
	rm -rf $B
