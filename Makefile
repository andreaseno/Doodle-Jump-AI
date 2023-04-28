run:
	@python3 doodle_gpt.py 1> log.log 2> log.log

clean:
	@rm *.txt
	@rm *.log