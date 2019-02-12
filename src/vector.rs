use rand::{Rng};
use rand::distributions::StandardNormal;

pub fn dot(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut acc = 0f64;
    for index in 0..a.len() {
        acc += a[index] as f64 * b[index] as f64;
    }
    return acc as f32;
}

//distance metric based on cosine distance which is offset from [-1,1] range into the [0,2] range
pub fn modified_cosine_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let d = dot(a, b);
    return (2f32 - (d + 1f32)).max(0f32);
}

pub fn random_unit_vector<R:Rng>(dimension:usize, rng: &mut R) -> Vec<f32> {

    //Generate a random vector
    let mut v : Vec<f32> = rng.sample_iter(&StandardNormal).take(dimension).map(|a| { a as f32 }).collect::<Vec<f32>>();

    //Calculate vector length
    let mut acc = 0f64;
    for i in 0..dimension {
        acc += v[i] as f64 * v[i] as f64;
    }
    let length = acc.sqrt();

    //Normalize
    for i in 0..dimension {
        v[i] /= length as f32;
    }

    return v;
}