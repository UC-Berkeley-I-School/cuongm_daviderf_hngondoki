# cuongm_daviderf_hngondoki_
Shared repo for submission of DATASCI207

# Background

Crafting the perfect playlist for a New Year’s party is an art, but what if data could make it a science? For a host who doesn’t dance often but knows the importance of energizing the crowd, the challenge is selecting tracks that will get guests moving. This project explores whether insights from Spotify's extensive music metrics, combined with advanced machine learning techniques, can predict how danceable a song is. By integrating streaming statistics, platform reach, and detailed audio features—such as tempo, energy, and valence—we aim to transform raw data into actionable insights. Leveraging the Spotify Web API, we enrich the dataset with song-specific attributes, creating a comprehensive foundation for our analysis. The ultimate goal is to design a predictive model that empowers anyone to compile data-driven, dance-friendly playlists. This endeavor bridges the worlds of music and machine learning, demonstrating how data can enhance even the most artistic of tasks.

# Data Source

The dataset, sourced from Kaggle, provides a detailed compilation of the most streamed Spotify songs of 2024, offering a wealth of information for analysis [^1]. It includes 4,600 songs with comprehensive insights into each track’s attributes, popularity, and presence across major music platforms such as Spotify, YouTube, TikTok, Apple Music, and more. Key features range from fundamental details like track name, album, artist, and release date to performance metrics such as streaming statistics, playlist reach, and social media engagement. Unique identifiers like the ISRC ensure data traceability, while platform-specific metrics like TikTok Likes, Spotify Streams, and Pandora Stations provide a multidimensional view of a song’s impact. Beyond streaming, the dataset captures qualitative aspects, such as explicit content tags and popularity scores, enabling a nuanced exploration of music trends. With its wide array of features, the dataset serves as a valuable resource for studying temporal trends, platform comparisons, artist impact, and cross-platform performance. This rich data foundation is pivotal for our project, as it offers the necessary variables to predict danceability and create an optimized playlist for any occasion.

# Literature Review 1

In her analysis in the notebook titled “Spotify Analysis Visualization”, Shahd Mohamed explored trends in the Spotify top hits dataset spanning two decades (2000–2019), providing valuable insights into song attributes and artist dominance [^2]. Using visualization techniques, she investigated patterns in audio features such as danceability, energy, tempo, and loudness, alongside metadata like popularity, year of release, and artist contributions. One of her key findings was the evolution of song duration and explicitness over time, which revealed shorter tracks and an increase in explicit content in recent years. Additionally, her work highlighted the dominance of a small subset of artists in producing top hits, with Rihanna, Drake, and Eminem leading in terms of track counts and average popularity scores.

The implications of Shahd Mohamed's findings extend to understanding consumer preferences and the music industry's strategic direction. The observed increase in explicit content may reflect shifting cultural norms, while the shortening of song durations aligns with contemporary consumption habits favoring bite-sized entertainment. Her study also underscores the role of audio features like energy and danceability in a track's popularity, offering predictive insights for artists and producers aiming to craft chart-topping music. By analyzing these data-driven trends, Shahd Mohamed's work contributes to a nuanced understanding of the interplay between artistic choices and audience preferences, a perspective that is integral for stakeholders in the music industry.

# Literature Review 2

Pragyan Tiwari's work on "Classifying Explicitness of Tracks Using LLAMA3-70B" offers a comprehensive approach to addressing inaccuracies within the "Most Streamed Spotify Songs 2024" dataset [^3]. Recognizing that the dataset's explicit content column was unreliable—where, for instance, tracks like Miley Cyrus' "Flowers" were incorrectly marked as explicit—Elgiriyewithana employed the LLAMA3-70B model to reassess and reclassify the explicitness of the songs. This reclassification involved leveraging the model's contextual understanding of lyrics and metadata to ensure greater precision in labeling. The refined dataset he produced aimed to resolve trust issues with the original dataset, particularly around explicitness labels, which have significant implications for curators, playlist algorithms, and industry analyses.

The findings underscore the importance of accurate data curation in large-scale datasets used for streaming analytics. Elgiriyewithana's work highlights not only the methodological rigor necessary for detecting and correcting dataset flaws but also the broader implications of erroneous classifications. Mislabeling explicit content can mislead users, compromise streaming platforms' credibility, and affect playlist recommendations and audience targeting. This initiative also demonstrates the potential of advanced language models like LLAMA3-70B in refining dataset quality, setting a precedent for future enhancements in similar datasets. His contribution ensures more reliable analyses and decisions drawn from the dataset while prompting further scrutiny into data reliability across music platforms.

# Using Spotify Web API to Retrieve Audio Features

The absence of audio feature data, such as "danceability," in the "Most Streamed Spotify Songs 2024" dataset presented a significant gap in the analysis of song characteristics and their suitability for dancing. To address this limitation, we leveraged Spotify's Web API, which provides detailed audio features for individual tracks via their unique Spotify IDs. This API offers attributes such as energy, valence, tempo, and instrumentalness, which are pivotal in evaluating the musical and perceptual qualities of a track. These features allow us to make informed predictions about a song's danceability, enabling a more nuanced understanding of what makes a track suitable for dancing.

Our approach involved first identifying and matching Spotify IDs for tracks in the dataset. Using the API, we sent requests to retrieve the audio features for each track in batches to optimize the process while adhering to rate limits. We ensured that missing IDs were handled efficiently through iterative searches, using track names and artist combinations to maximize retrieval success. Ultimately, the integration of Spotify Web API data enhanced the dataset's analytical potential, bridging the gap between raw streaming statistics and the perceptual qualities of music. This enriched dataset provides a robust foundation for predicting danceability and analyzing trends across musical genres and platforms.

# Feature Selection

For our analysis and model to predict the danceability of songs, we carefully selected features from the dataset that are most relevant and quantifiable. The features were chosen based on their data type (int64 and float64) to ensure compatibility with machine learning algorithms, resulting in a streamlined set of predictors. These selected X-variables represent key song attributes and metadata that can potentially influence a track’s danceability:

- **artist_count**: The number of artists contributing to a track, which can indicate collaborations or diversity in musical style.
- **released_year, released_month, released_day**: The release date components that may reflect evolving musical trends or temporal patterns in danceable music.
- **in_spotify_playlists**: The number of Spotify playlists that include the track, serving as a proxy for its general popularity and exposure.
- **in_spotify_charts**: The track’s presence in Spotify’s charts, highlighting its ranking and audience engagement.
- **in_apple_playlists**: The number of Apple Music playlists featuring the track, capturing its cross-platform reach.
- **in_apple_charts**: The ranking of the track on Apple Music, representing its reception on that platform.
- **in_deezer_charts**: The presence and ranking of the track on Deezer, adding another dimension to its multi-platform performance.
- **bpm (beats per minute)**: A measure of tempo that directly affects a track’s rhythmic appeal and suitability for dancing.
- **valence_%**: A percentage score reflecting the positivity or happiness conveyed by the track.
- **energy_%**: A percentage measure of the intensity and activity level in the track, with higher scores indicating faster, louder, or more energetic songs.
- **acousticness_%**: A percentage confidence measure of the track’s acoustic properties, with higher values indicating more acoustic content.
- **instrumentalness_%**: A percentage likelihood that the track contains minimal or no vocals, distinguishing instrumental compositions.
- **liveness_%**: A measure of the likelihood that the track was performed live, capturing the presence of an audience.
- **speechiness_%**: A percentage measure of the amount of spoken word in the track, which can impact its rhythm and lyrical style.

# Outcome Variable

The target Y-variable for our model is **danceability**, which was originally defined as a continuous float between 0 and 1. For better interpretability and classification purposes, we re-categorized this variable into three distinct classes: Low, Medium, and High danceability. This transformation allows us to analyze not only whether a track is danceable but also the degree to which it is suitable for dancing. This feature engineering step provides a more practical framework for predicting and understanding danceability trends in music.

By focusing on these features, we aim to build a robust model that captures the essential attributes influencing danceability while maintaining interpretability and practical relevance for music analysts and enthusiasts.

# Footnotes

[^1]: Nidula Elgiriyewithana, “Most Streamed Spotify Songs 2024,” Kaggle, https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024/data.
[^2]: Shahd Mohamed, “Spotify Analysis Visualization,” Kaggle, https://www.kaggle.com/code/shahdmohamed1/spotify-analysis-visualization.
[^3]: Pragyan Tiwari, “Classifying Explicitness of Tracks Using LLAMA3-70B,” Kaggle, https://www.kaggle.com/code/pragyantiwari/classifying-explicity-of-tracks-using-llama3-70b.
