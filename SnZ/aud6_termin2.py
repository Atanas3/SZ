critics={
    'Lisa Rose': {'Catch Me If You Can': 3.0 , 'Snakes on a Plane': 3.5, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0, 'Snitch': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5,  'The Night Listener': 3.0,'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Catch Me If You Can': 2.5, 'Lady in the Water': 2.5,'Superman Returns': 3.5, 'The Night Listener': 4.0, 'Snitch': 2.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,'The Night Listener': 4.5, 'Superman Returns': 4.0,'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,'Just My Luck': 2.0, 'Superman Returns': 3.0, 'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Catch Me If You Can': 4.5, 'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5, 'Snitch': 4.5},
    'Toby': {'Snakes on a Plane':4.5, 'Snitch': 5.0},
	'Michelle Nichols': {'Just My Luck' : 1.0, 'The Night Listener': 4.5, 'You, Me and Dupree': 3.5, 'Catch Me If You Can': 2.5, 'Snakes on a Plane': 3.0},
	'Gary Coleman': {'Lady in the Water': 1.0, 'Catch Me If You Can': 1.5, 'Superman Returns': 1.5, 'You, Me and Dupree': 2.0},
	'Larry': {'Lady in the Water': 3.0, 'Just My Luck': 3.5, 'Snitch': 1.5, 'The Night Listener': 3.5}
	}

if __name__ == '__main__':
    from recommender_systems import sim_distance, sim_pearson,\
        top_matches, get_recommendations, transform_prefs, \
        get_recommendations_item_based

    print(sim_distance(critics, 'Lisa Rose', 'Toby'))
    print(sim_pearson(critics, 'Lisa Rose', 'Toby'))

    print(top_matches(critics, 'Toby', 2, sim_distance))
    print(top_matches(critics, 'Toby', 2, sim_pearson))

    print(get_recommendations(critics, 'Toby'))
    print(get_recommendations(critics, 'Toby', sim_distance))

    movies = transform_prefs(critics)

    print(movies)

    print(top_matches(movies, 'Superman Returns', 2))
    print(top_matches(movies, 'Superman Returns', 2, sim_distance))

    print(get_recommendations_item_based(movies, 'Toby'))


