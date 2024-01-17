all:
	@dune exec ./test.exe
	@dune exec ./char_rnn.exe

square:
	@dune exec ./square.exe

ngram:
	@dune exec ./ngram.exe

shakespeare.txt:
	wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O $@

miserables.txt:
	rm -f $@
	wget https://www.gutenberg.org/cache/epub/17489/pg17489.txt -O - >> $@
	wget https://www.gutenberg.org/cache/epub/17489/pg17489.txt -O - >> $@
	wget https://www.gutenberg.org/cache/epub/17494/pg17494.txt -O - >> $@
	wget https://www.gutenberg.org/cache/epub/17494/pg17494.txt -O - >> $@
	wget https://www.gutenberg.org/cache/epub/17519/pg17519.txt -O - >> $@
