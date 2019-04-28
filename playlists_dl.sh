# get the remote playlists
gsutil ls gs://titan-source-dinar-raw/playlist > remote_playlists.txt

wc -l ./remote_playlists.txt

# get the local playlists
ls ./playlist > local_playlists.txt

wc -l ./local_playlists.txt

# add slashes at the begining and the end of each line
sed -i "s/^/\//" local_playlists.txt
sed -i "s/$/\//" local_playlists.txt

# get the playlists we want to keep
grep -vF -f local_playlists.txt remote_playlists.txt > playlists_toDL.sh

wc -l playlists_toDL.sh

# change file playlists_toDL in order to dl each playlist we do not have locally
sed -i "s/^/gsutil cp -r /" playlists_toDL.sh
sed -i "s/$/ .\/playlist/" playlists_toDL.sh

# run the playlists_toDL bash to dl 
sh ./playlists_toDL.sh

# remove files
rm remote_playlists.txt
rm local_playlists.txt
rm playlists_toDL.sh

