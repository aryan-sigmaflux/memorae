# install docker + compose plugin (once), then:
cd ~/memorae
cp .env.example .env    
# fill in your real secrets + change the passwords
docker compose up -d --build


cd ~/memorae && git pull && docker compose up -d --build


docker compose logs -f app
# look for "Starting Memorae", no MinIO/db errors
