# makefile uses tabs not spaces, if you use spaces make won't work. 
yamda-dock:
	sudo docker-compose build --no-cache --force-rm --pull yamda-dock && sudo docker tag "yamda-dock:latest" `date +%Y%m%d%H%M%S -u`;

cleanup:
	-sudo docker-compose rm -f
	-sudo docker images purge
	-sudo docker system prune -fa
