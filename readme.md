## Hypernonsense

[![](https://meritbadge.herokuapp.com/hypernonsense)](https://crates.io/crates/hypernonsense)

An implementation of [Locality Sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) with [random projection](https://en.wikipedia.org/wiki/Random_projection).

Locality sensitive hashing allows you to query a very large set of points in extremely high dimensional space to find the nearest points to a query point.

This is useful when working with [word vectors](https://en.wikipedia.org/wiki/Word_embedding) where the [cosine distance](https://en.wikipedia.org/wiki/Cosine_distance) between words is the similarity of those words. You can use a hypernonsense to find the most similar words to a given point in word space.

## Usage

Hypernonsense contains two types of index. The basic `hyperindex` is very fast to generate results but is fairly inaccurate. The `multiindex` contains more than one `hyperindex` and aggregates the results to improve accuracy.

#### hyperindex

```
// We're working in 300 dimensional space
const dimension : usize = 300;

// How many planes should the space be split by. More planes increases speed but decreases accuracy
// A good starting value is: 2 ^ planes * 10 == number of points
const planes : u8 = 10;

// Create the index
let mut index = HyperIndex::new(dimension, planes, &mut thread_rng());

// Populate the index with random data
let mut rng = thread_rng();
for key in 0..10000usize {

    // Generate a random vector
    let v = random_unit_vector(dimension, &mut rng);
    
    // The `key` can be any type - When you query the hyperindex you will get back a set of keys. In this case we'll just use the index.    
    vectors.push((key, v));
}

// Find approximately the nearest vectors to a random query vector. The key we used was `i32` so we get back a `Vec<i32>`
let result : Option<&Vec<i32>> = index.group(random_unit_vector(dimension, &mut rng));
```

Result from the `hyperindex` will be very fast to retrieve - it's just testing the query point against the `10` (in this example) hyperplanes and retrieving a pre-populated list. However the quality of results will be very poor, points are prearranged into groups and any points which lie near to a hyperplane will not retrieve all of the closest points.

#### multiindex

```
// We're working in 300 dimensional space
const dimension : usize = 300;

// How many sub-indices should there be. More indices increases accuracy, but decreases speed and increases memory consumption.
const indices : u8 = 10

// How many planes should the space be split by. More planes increases speed but decreases accuracy
// A good starting value is: 2 ^ planes * 10 == number of points
const planes : u8 = 10;

// Create the index
let mut index = MultiIndex::new(dimension, 10, 10, &mut thread_rng());

// Populate index
// ...exactly the same as the hyperindex example

// Find the approximately nearest vectors to a random query vector. The key we used was `i32` so we get back a `Vec<DistanceNode<i32>>`
// This require use to supply the number of points we want and a distance metric to choose them by
const nearest_count : usize = 100;
let result : Vec<DistanceNode<i32>> = index.nearest(&random_unit_vector(dimension, &mut rng), nearest_count, |point, key| {
    return distance(point, get_vector_by_key(key));
});
```

The `multiindex` contains within in a number of `hyperindex` (specified as `10` in this example). When queried it fetches the nearest group from all sub-indices and then merges them together, keeping the closest points (limited to `100` in this example) according to the distance metric (`distance` function in this example).

## Tweaking Parameter

When using this you must be aware that it is a probabilistic data structure - results that it returns are approximately correct. You should experiment with the two parameters until you achieve a level of speed and accuracy that you are happy with.

#### Planes

The `hyperindex` is split into regions of space between the random hyperplanes it generates, when you query the index you simply get back a list of points in the same region.

If you don't have very many hyperplanes you will get back very large result sets and the index isn't really helping you. The limit of this is zero hyperplanes, in which case every query will return the entire set!

If you have too many hyperplanes each group will have a very small number of points in it and the quality of results will suffer as it becomes increasingly likely that the correct answer will have been classified into a different group. Once `2 ^ planes >= data_points` it is likely that every point will be in a different group and the index is almost useless.

#### Sub Index Count

The `multiindex` contains multiple `hyperindex` each split with the same number of planes, but with different (randomised) plane orientations. When you query the `multiindex` it queries all the sub-indices and merges the results together. This approach increases the accuracy of the query by taking the best results (according to the distance metric) from each sub query. Increasing the sub index count increases memory consumption and query time.

If you don't have very many sub-indices you will basically just be querying a `hyperindex` and the distance metric will be useless.