use mnist::MnistBuilder;
use ndarray::Array2;

pub struct MnistData {
    pub trn_img: Array2<f64>,
    pub trn_lbl: Array2<f64>,
    pub tst_img: Array2<f64>,
    pub tst_lbl: Array2<f64>,
}

fn vec_to_array_img(vec: Vec<u8>, size: usize) -> Array2<f64> {
    let img = Array2::from_shape_vec((size, 28 * 28), vec).unwrap();
    img.map(|x| *x as f64 / 255.0)
}

fn vec_to_array_lbl(vec: Vec<u8>, size: usize) -> Array2<f64> {
    let mut lbl = Array2::zeros((size, 10));
    for i in 0..size {
        lbl[[i, vec[i] as usize]] = 1.0;
    }
    lbl
}

impl MnistData {
    pub fn new() -> Self {
        let mnist = MnistBuilder::new().label_format_digit().finalize();

        let trn_img = mnist.trn_img;
        let trn_lbl = mnist.trn_lbl;
        let tst_img = mnist.tst_img;
        let tst_lbl = mnist.tst_lbl;

        let trn_img = vec_to_array_img(trn_img, 60_000);
        let trn_lbl = vec_to_array_lbl(trn_lbl, 60_000);
        let tst_img = vec_to_array_img(tst_img, 10_000);
        let tst_lbl = vec_to_array_lbl(tst_lbl, 10_000);

        MnistData {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
        }
    }
}
