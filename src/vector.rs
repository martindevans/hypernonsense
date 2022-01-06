use rand::{Rng};
use rand_distr::StandardNormal;

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut acc = 0f32;
    for index in 0..a.len() {
        acc += a[index] * b[index];
    }

    return acc;
}

//distance metric based on cosine distance which is offset from [-1,1] range into the [0,2] range
pub fn modified_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    (2f32 - (d + 1f32)).max(0f32)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut acc = 0f32;
    for index in 0..a.len() {
        acc += (a[index] - b[index]).powf(2f32);
    }
    acc = acc.sqrt();

    return acc;
}

pub fn random_unit_vector<R:Rng>(dimension:usize, rng: &mut R) -> Vec<f32>
{
    // Generate a random vector
    let mut v : Vec<f32> = rng.sample_iter(&StandardNormal).take(dimension).collect::<Vec<f32>>();

    // Calculate vector length
    let length = v.iter()
        .map(|a| *a as f64)
        .map(|a| a * a)
        .sum::<f64>()
        .sqrt();

    // Normalize
    for i in 0..dimension {
        v[i] /= length as f32;
    }

    v
}