run:
	@python3 DoodlePPO.py 1> log.log 2> log.log

clean:
	@rm *.txt
	@rm *.log