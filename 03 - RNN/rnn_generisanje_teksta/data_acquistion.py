from selenium import webdriver
from fuzzywuzzy import fuzz

driver = webdriver.Firefox()

# all lyrics downloaded from tekstovi.net

driver.get("https://tekstovi.net")

target_artists = {
    'folk': ['ceca', 'aca lukas', 'lepa brena', 'dara bubamara', 'seka aleksic', 'dragana mirkovic', 'mile kitic', 'mitar miric', 'toma zdravkovic', 'lepa lukic', 'silvana armenulic', 'semsa suljakovic', 'ahmedovski', 'indira radic', 'sasa matic'],
    'rock': ['generacija 5', 'kerber', 'bijelo dugme', 'azra', 'bajaga', 'galija', 'parni valjak', 'riblja corba'],
    'pop': ['funky g', 'jelena tomasevic', 'marija serifovic', 'jelena karleusa', 'zeljko joksimovic', 'severina', 'bekvalac', 'sasa kovacevic', 'luna']
}

first_letter_links = [link.get_attribute("href") for link in
                      driver.find_elements_by_xpath("//*[@id='meni_slova']//td/a")]

artists = {}
for first_letter_link in first_letter_links:
    driver.get(first_letter_link)
    for artist in driver.find_elements_by_xpath("//*[@class='artLyrList']/a"):
        artists[artist.text] = artist.get_attribute('href')

# scrape lyrics for all desired artists
for genre in target_artists.keys():
    for artist_name in target_artists[genre]:
        for known_name in artists.keys():
            if fuzz.ratio(artist_name.lower(), known_name.lower()) > 95:
                # open artist page and scrape all songs
                print("Started: " + known_name)
                driver.get(artists[known_name])
                song_links = [link.get_attribute('href') for link in
                              driver.find_elements_by_xpath("//*[@class='artLyrList']/a")]
                for song_link in song_links:
                    driver.get(song_link)
                    song_name = driver.find_element_by_xpath("//h2[@class='lyricCapt']").text
                    lyric_text = ""
                    for lyric_part in driver.find_elements_by_xpath("//*[@class='lyric']"):
                        lyric_text += lyric_part.text + '\n\n'
                    with open('./data/'+genre+'/'+known_name+' - '+song_name+'.txt', 'w') as output_stream:
                        output_stream.write(lyric_text)
                print("Finished: "+known_name)
driver.close()
