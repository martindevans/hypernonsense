use std::collections::HashMap;

use rayon::prelude::*;
use rand::{Rng};
use bit_vec::{BitVec};

use crate::vector::{ dot, random_unit_vector };

pub struct HyperIndex<K:Send> {
    planes: Vec<Vec<f32>>,
    groups: HashMap<BitVec, Vec<K>>,
    dims: usize
}

impl<K:Send> HyperIndex<K> {
    pub fn new<R : Rng + Sized>(dimension: usize, hyperplane_count: u8, mut rng: &mut R) -> HyperIndex<K>
    {
        let mut planes = Vec::<Vec<f32>>::with_capacity(hyperplane_count as usize);
        for _ in 0..hyperplane_count {
            planes.push(random_unit_vector(dimension, &mut rng));
        }

        return HyperIndex {
            planes: planes,
            groups: HashMap::new(),
            dims: dimension
        }
    }

    pub fn dimensions(&self) -> usize {
        return self.dims;
    }

    pub fn groups_len(&self) -> usize {
        return self.groups.len();
    }

    pub fn planes_len(&self) -> usize {
        return self.planes.len();
    }

    pub fn key(&self, vector: &Vec<f32>) -> BitVec {

        let mut key = BitVec::with_capacity(self.planes.len());

        let bits:Vec<bool> = self.planes.par_iter().map(|plane| {
            let d = dot(plane, vector);
            return d > 0f32;
        }).collect();

        for bit in bits.iter() {
            key.push(*bit);
        }
        
        return key;
    }

    pub fn add(&mut self, key: K, vector: &Vec<f32>) {

        //Build bit vector, each bit indicates which side of the hyperplane the point is on
        let bits = self.key(&vector);

        //Insert this item into the appropriate group
        self.groups
            .entry(bits)
            .or_insert(Vec::new())
            .push(key);
    }

    pub fn group(&self, key: &BitVec) -> Option<&Vec<K>> {
        return self.groups.get(&key);
    }
}

#[cfg(test)]
mod tests
{
    use rand::prelude::*;

    use crate::hyperindex::{ HyperIndex };
    use crate::vector::{ random_unit_vector, modified_cosine_distance };

    #[test]
    fn new_creates_index<'a>() {
        let a = HyperIndex::<usize>::new(300, 10, &mut thread_rng());

        assert_eq!(300, a.dimensions());
        assert_eq!(0, a.groups_len());
        assert_eq!(10, a.planes_len());
    }

    #[test]
    fn add_adds_points<'a>() {
        let mut a = HyperIndex::new(300, 10, &mut thread_rng());

        let v = random_unit_vector(300, &mut thread_rng());
        a.add(0, &v);
    }

    #[test]
    fn it_works<'a>() {
        let mut a = HyperIndex::new(300, 10, &mut thread_rng());

        let mut vectors = Vec::new();

        let mut rng = thread_rng();
        for key in 0..10000usize {
            let v = random_unit_vector(300, &mut rng);
            a.add(key, &v);
            vectors.push((key, v));
        }

        println!("Groups:{:?}", a.groups_len());

        //Get closest points from simple search through the entire set
        println!();
        println!("Linear results:");
        let query_point = vectors[0].clone();
        let mut nearest_linear: Vec<(f32, &(usize, Vec<f32>))> = vectors.iter().map(|item| (modified_cosine_distance(&item.1, &query_point.1), item)).collect();
        nearest_linear.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..20 {
            println!("idx:{:?}\t\tdist:{:?}", (nearest_linear[i].1).0, nearest_linear[i].0);
        }
            
        //Use the index
        println!();
        println!("Index results:");
        let near = a.group(&a.key(&query_point.1));
        if near.is_none() {
            panic!();
        }
        let near = near.unwrap();

        let mut results: Vec<(f32, &(usize, Vec<f32>))> = near.iter().map(|i| &vectors[*i]).map(|item| (modified_cosine_distance(&item.1, &query_point.1), item)).collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0.. results.len().min(20) {
            println!("idx:{:?}\t\tdist:{:?}", (results[i].1).0, results[i].0);
        }
    }
}