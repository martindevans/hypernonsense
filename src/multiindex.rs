use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

use rayon::prelude::*;
use rand::{Rng};
use bit_vec::{BitVec};

use crate::hyperindex::{ HyperIndex };

pub struct DistanceNode<K> {
    pub key: K,
    pub distance: f32
}

pub struct MultiIndex<K:Send> {
    indices: Vec<HyperIndex<K>>
}

impl<K:Clone+Eq+Hash+Debug+Send> MultiIndex<K> {
    pub fn new<R : Rng + Sized>(dimension: usize, index_count: u8, hyperplane_count: u8, mut rng: &mut R) -> MultiIndex<K>
    {
        return MultiIndex {
            indices: (0..index_count).map(|_| HyperIndex::new(dimension, hyperplane_count, &mut rng)).collect()
        }
    }

    fn merge(mut a: Vec<DistanceNode<K>>, mut b: Vec<DistanceNode<K>>, limit: usize) -> Vec<DistanceNode<K>> {

        //Add all items from B into A
        a.append(&mut b);

        //Sort merged lists by distance
        a.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        //Trim to size
        a.truncate(limit);

        return a;
    }

    fn merge_key_results<F:Fn(&Vec<f32>, &K) -> f32>(point: &Vec<f32>, index: &HyperIndex<K>, key: &BitVec, get_dist: &F, found: &mut HashSet<K>, results: Vec<DistanceNode<K>>, count: usize) -> Vec<DistanceNode<K>> {
        if let Some(g) = index.group(&key) {

            let candidates:Vec<DistanceNode<K>> = g.iter().filter_map(|key| {

                //skip items which we've already found.
                //They're either already in the result list, or were already rejected as too far away.
                if !found.insert(key.clone()) {
                    return None;
                }

                return Some(DistanceNode {
                    key: key.clone(),
                    distance: get_dist(&point, &key)
                });
            }).collect();

            return MultiIndex::<K>::merge(results, candidates, count);
        }

        return results;
    }

    pub fn nearest<F:Fn(&Vec<f32>, &K) -> f32>(&self, point: &Vec<f32>, count: usize, get_dist: F) -> Vec<DistanceNode<K>> {
        let mut results = Vec::with_capacity(count);
        let mut found = HashSet::new();

        for index in self.indices.iter() {
            let k = index.key(&point);
            results = MultiIndex::<K>::merge_key_results(&point, &index, &k, &get_dist, &mut found, results, count);

            //Fetch all adjacent keys (all keys different by just one bit)
            for i in 0..k.len() {
                let mut k2 = k.clone();
                k2.set(i, !k2.get(i).unwrap());

                results = MultiIndex::<K>::merge_key_results(&point, &index, &k2, &get_dist, &mut found, results, count);
            }
        }

        return results;
    }

    pub fn add(&mut self, key: K, vector: &Vec<f32>) {

        // Build a list of work that needs doing (tuple of index, key and vector)
        let mut work:Vec<_> = self.indices.iter_mut()
            .map(|idx| (idx, &key, vector))
            .collect();

        //Add items in parallel
        work.iter_mut().for_each(|work| {
            work.0.add(work.1.clone(), work.2);
        });
    }
}

#[cfg(test)]
mod tests
{
    use rand::prelude::*;
    use std::collections::HashSet;

    extern crate time;
    use time::PreciseTime;

    use crate::multiindex::{ MultiIndex };
    use crate::vector::{ random_unit_vector, modified_cosine_distance };

    #[test]
    fn new_creates_index<'a>() {
        let a = MultiIndex::<usize>::new(300, 10, 10, &mut thread_rng());

        //assert_eq!(300, a.dimensions());
        //assert_eq!(0, a.groups_len());
        //assert_eq!(10, a.planes_len());
    }

    #[test]
    fn multiindex_compare<'a>() {
        let mut a = MultiIndex::new(300, 10, 10, &mut thread_rng());

        let mut vectors = Vec::new();

        let mut rng = thread_rng();
        for key in 0..10000usize {
            let v = random_unit_vector(300, &mut rng);
            a.add(key, &v);
            vectors.push((key, v));
        }

        let start_linear = PreciseTime::now();

        //Get closest points from simple search through the entire set
        println!();
        println!("Linear results:");
        let query_point = vectors[0].clone();
        let mut nearest_linear: Vec<(f32, &(usize, Vec<f32>))> = vectors.iter().map(|item| (modified_cosine_distance(&item.1, &query_point.1), item)).collect();
        nearest_linear.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let end_linear = PreciseTime::now();
        println!("{} seconds for linear", start_linear.to(end_linear));

        for i in 0..20 {
            println!("idx:{:?}\t\tdist:{:?}", (nearest_linear[i].1).0, nearest_linear[i].0);
        }

        let start_indexed = PreciseTime::now();

        //Use the index
        println!();
        println!("Index results:");
        let near = a.nearest(&query_point.1, 100, |p, k| {
            return modified_cosine_distance(p, &vectors[*k].1);
        });

        let end_indexed = PreciseTime::now();
        println!("{} seconds for index", start_indexed.to(end_indexed));
        
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