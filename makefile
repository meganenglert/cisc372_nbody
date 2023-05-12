FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody 