all:
	make -C aurora-v4/aurora-v4/
	make -C affix-base/
	make -C aurora-v4/alg-test/
	
clean:
	make clean -C aurora-v4/aurora-v4/
	make clean -C affix-base/
	make clean -C aurora-v4/alg-test/
	
