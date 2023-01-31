use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

use rand::Rng;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::hyperindex::HyperIndex;

pub struct DistanceNode<K: Eq+Hash> {
    pub key: K,
    pub distance: f32
}

impl<K:Eq+Hash> Eq for DistanceNode<K>
{
}

impl<K:Eq+Hash> PartialEq for DistanceNode<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K:Eq+Hash> Hash for DistanceNode<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

pub struct MultiIndex<K:Send> {
    indices: Vec<HyperIndex<K>>
}

impl<K:Clone+Eq+Hash+Debug+Send> MultiIndex<K> {
    pub fn new<R : Rng + Sized>(dimension: usize, index_count: u8, hyperplane_count: u8, mut rng: &mut R) -> MultiIndex<K> {
        MultiIndex {
            indices: (0..index_count).map(|_| HyperIndex::new(dimension, hyperplane_count, &mut rng)).collect()
        }
    }

    pub fn nearest<F:Fn(&Vec<f32>, &K) -> f32>(&self, point: &Vec<f32>, count: usize, get_dist: F) -> Vec<DistanceNode<K>>
    {
        // Collect one group of results from every hyperindex
        // Dedupe by collecting into an intermediate hashset
        let mut results: Vec<_> = self.indices.iter()
            .flat_map(|i| i.group(&i.key(&point)))
            .flat_map(|a| a)
            .map(|a| DistanceNode { distance: get_dist(point, a), key: a.clone() })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        // Sort into distance order and truncate to length
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(count);

        return results;
    }

    pub fn add(&mut self, key: K, vector: &Vec<f32>)
    {
        // Build a list of work that needs doing (tuple of index, key and vector)
        let mut work:Vec<_> = self.indices.iter_mut()
            .map(|idx| (idx, key.clone(), vector))
            .collect();

        //Add items in parallel
        work.par_iter_mut().for_each(|work| {
            work.0.add(work.1.clone(), work.2);
        });
    }

    pub fn dimensions(&self) -> usize {
        self.indices[0].dimensions()
    }

    pub fn planes_len(&self) -> usize {
        self.indices[0].planes_len()
    }

    pub fn indices_len(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests
{
    use rand::prelude::*;
    use std::collections::HashSet;

    extern crate time;
    use time::Instant;

    use crate::multiindex::MultiIndex;
    use crate::vector::{ random_unit_vector, euclidean_distance };

    #[test]
    fn new_creates_index() {
        let a = MultiIndex::<usize>::new(300, 15, 10, &mut thread_rng());

        assert_eq!(300, a.dimensions());
        assert_eq!(10, a.planes_len());
        assert_eq!(15, a.indices_len());
    }

    #[test]
    fn multiindex_compare() {
        let mut a = MultiIndex::new(300, 30, 5, &mut thread_rng());

        let mut vectors = Vec::new();

        let mut rng = thread_rng();
        for key in 0..10000usize {
            let v = random_unit_vector(300, &mut rng);
            a.add(key, &v);
            vectors.push((key, v));
        }

        let start_linear = Instant::now();

        //Get closest points from simple search through the entire set
        println!();
        println!("Linear results:");
        let query_point = vectors[0].clone();
        let mut nearest_linear: Vec<(f32, &(usize, Vec<f32>))> = vectors.iter().map(|item| (euclidean_distance(&item.1, &query_point.1), item)).collect();
        nearest_linear.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let end_linear = Instant::now();
        println!("{:?} seconds for linear", end_linear - start_linear);

        for i in 0..20 {
            println!("idx:{:?}\t\tdist:{:?}", (nearest_linear[i].1).0, nearest_linear[i].0);
        }

        let start_indexed = Instant::now();

        //Use the index
        println!();
        println!("Index results:");
        let near = a.nearest(&query_point.1, 100, |p, k| {
            euclidean_distance(p, &vectors[*k].1)
        });

        let end_indexed = Instant::now();
        println!("{:?} seconds for index", end_indexed - start_indexed);
        
        for i in 0.. near.len().min(20) {
            println!("idx:{:?}\t\tdist:{:?}", near[i].key, near[i].distance);
        }

        let linear_set: HashSet<_> = nearest_linear.iter().map(|a| (a.1).0).take(20).collect();
        let near_set: HashSet<_> = near.iter().map(|a| a.key).take(20).collect();
        let overlap: Vec<_> = linear_set.intersection(&near_set).collect();
        println!();
        println!("Overlap:{:?}/20", overlap.len())
    }
}