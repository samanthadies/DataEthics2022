JARS = \
bsh.jar \
hppc-0.7.1.jar \
junit-4.12.jar \
commons-math3-3.6.1.jar

# note: jython.jar not included above, because only needed by bin/mallethon

unpack: $(JARS)
	rm -rf $@
	mkdir $@
	for f in $(JARS) ; do \
	  (cd $@ ; jar -xf ../$$f) ; \
	done
	rm -rf $@/META-INF

mallet-deps.jar: unpack
	cd $< ; \
	jar -cf ../$@ *

clean:
	rm -rf unpack
	rm -f mallet-deps.jar
