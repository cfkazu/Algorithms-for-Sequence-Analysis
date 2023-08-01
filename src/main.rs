pub mod s_algo{
    //! string algorithm without using FFT
    use std::collections::HashMap;
    
    /// ブルートフォースアルゴリズムでパターンマッチングを行う
    pub fn brute_search(text:&str,pattern:&str)->Vec<usize>{
        let mut res:Vec<usize> = vec![];
        for i in 0..text.len(){
            if text[i..].starts_with(pattern){
                res.push(i);
            }   
        }
        res
    }
    /// shift-andアルゴリズムでパターンマッチングを行う．128文字以降はブルートフォースアルゴリズムを使用する
    pub fn new_shift_and_brute(text:&str,pattern: &str)->Vec<usize>{
        let pattern_sub = if pattern.len() > 128 {pattern[..128].to_string()}else{pattern.to_string()};
        let res = new_shift_and(&text,&pattern_sub);
        let mut ans:Vec<usize> = vec![];
        for index in res.iter(){
            if text[*index..].starts_with(pattern){
                ans.push(*index);
            }
        }
        ans
    }
    /// shift-andアルゴリズムでパターンマッチングを行う．128文字までしか対応していない．
    pub fn new_shift_and(text:&str,pattern: &str)->Vec<usize>{
        let pattern_len = pattern.len();
        let mut ans:Vec<usize> = vec![];
        let vocab = vec!['a','b','c','d','A','B','C','D'];
        let masks = make_mask(pattern,&vocab,false);
        let text_chars = text.chars().collect::<Vec<char>>();

        let mut now:u128 = 0;
        for i in 0.. text_chars.len(){
            now = ((now<<1)|1)&masks[&text_chars[i]];
            if now & (1<<(pattern_len-1)) != 0{
                ans.push(i+1-pattern_len);
            }
        }
        ans
    }

    /// shift-andアルゴリズムで用いるマスクを作成する
    pub fn make_mask(pattern: &str,vocab:&Vec<char>,neg:bool)->HashMap<char,u128>{
        let mut mask = vec![0_u128;vocab.len()];
        for i in 0..pattern.len(){
            mask[vocab.iter().position(|&x| x == pattern[i..].chars().next().unwrap()).unwrap()] |= 1<<i;
        }
        let mut res = HashMap::new();
        for i in 0..vocab.len(){
            if neg{
                res.insert(vocab[i],!mask[i]);
            }else{
                res.insert(vocab[i],mask[i]);
            }
        }
        res
    }
}

pub mod make_test{
    //! make test datasets
    use std::{io::Write, fs};
    use rand::Rng;
    use crate::s_algo::brute_search;

    /// パターンが本文の部分文字列になっているテストデータを作成する
    pub fn make_substring(len:usize,pattern_num:usize,num:usize,types:usize){
        fs::create_dir(format!("tests/tests_substring{}_{}_{}types",len,pattern_num,types));
        for i in 0..num{
            let mut file = std::fs::File::create(format!("tests/tests_substring{}_{}_{}types/{}.txt",len,pattern_num,types,i)).unwrap();
            let mut rng = rand::thread_rng();
            let mut s = String::new();
            for _ in 0..len{
                let c = rng.gen_range(0..types);
                s.push((c+97) as u8 as char);
            }
            let mut pattern = s[0..pattern_num].to_string();
            write!(file,"{}\n",s).unwrap();
            write!(file,"{}\n",pattern).unwrap();
            let ans = brute_search(&s, &pattern);
            let ans  = ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",");
            write!(file,"{}\n",ans).unwrap();
        }
    }
    ///パターンも本文もランダムなテストデータを作成する
    pub fn make_random(len:usize,pattern_num:usize,num:usize,types:usize){
        fs::create_dir(format!("tests/tests_random{}_{}_{}types",len,pattern_num,types));
        for i in 0..num{
            println!("now:{}",i);
            let mut file = std::fs::File::create(format!("tests/tests_random{}_{}_{}types/{}.txt",len,pattern_num,types,i)).unwrap();
            let mut rng = rand::thread_rng();
            let mut s = String::new();
            for _ in 0..len{
                let c = rng.gen_range(0..types);
                s.push((c+97) as u8 as char);
            }
            let mut pattern = String::new();
            for _ in 0..pattern_num{
                let c = rng.gen_range(0..types);
                pattern.push((c+97) as u8 as char);
            }
            write!(file,"{}\n",s).unwrap();
            write!(file,"{}\n",pattern).unwrap();
            let ans = brute_search(&s, &pattern);
            let ans  = ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",");
            write!(file,"{}\n",ans).unwrap();
        }
    }
}
pub mod my_corr{
    //! correlation using FFT
    use rustfft::{Fft,num_complex::Complex,FftDirection, algorithm::Radix4};

    pub fn correlation(a:Vec<f64>,b:Vec<f64>)->Vec<f64>{
        let (n,m) = (a.len(),b.len());
        let s = n+m-1;
        let t = s.next_power_of_two();
        let fft = Radix4::<f64>::new(t, FftDirection::Forward);
        let mut fa = a.iter().chain(std::iter::repeat(&0.0)).take(t).map(|&x| Complex{re:x,im:0.0}).collect::<Vec<_>>();
        let mut fb = b.iter().chain(std::iter::repeat(&0.0)).take(t).map(|&x| Complex{re:x,im:0.0}).collect::<Vec<_>>();
        fft.process(&mut fa);
        fft.process(&mut fb);
        for i in 0..t{
            fa[i] = fa[i]*fb[i].conj();
        }
        let fft = Radix4::new(t, FftDirection::Inverse);
        fft.process(&mut fa);
        (0..(n+m-1)).into_iter().map(|i| fa[i].re).collect::<Vec<_>>()
    }
    pub fn correlation_usize(a: &[usize], b: &[usize]) -> Vec<usize> {
        let a: Vec<f64> = a.iter().map(|&a| a as f64).collect();
        let b: Vec<f64> = b.iter().map(|&b| b as f64).collect();
        let s = a.len() + b.len() - 1;
        let t = s.next_power_of_two();
        let c = correlation(a,b);
        c.iter().map(|&z| z.round() as usize / t).collect()
    }
}
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
macro_rules! measure {
  ($x:expr) => {
    {
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      println!("{}.{:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
      (result,end)
    }
  };
}
/// ハミング距離をFFTを用いて計算する
fn calc_hamming(text:&str,pattern:&str)->Vec<usize>{
    let mut vocabulary = pattern.chars().collect::<Vec<char>>();
    vocabulary.sort();
    vocabulary.dedup();
    let mut res = vec![0_usize;text.len()+pattern.len()-1];
    for c in vocabulary.iter(){
        let text_vec_c = convert_to_usize_vec(text,*c);
        let pattern_vec_c = convert_to_usize_vec(pattern,*c);
        let c = my_corr::correlation_usize(&text_vec_c,& pattern_vec_c);
        for i in 0..c.len(){
            res[i]+=c[i];
        }
    }
    res
}
/// 文字列をベクトルに変換する
fn convert_to_usize_vec(s: &str,target:char) -> Vec<usize> {
    s.chars().map(|c| if c == target || c == '?' {1}else{0}).collect()
}
///　ハミング距離からパターンが出現する場所を求める
fn hamming_to_placeid(hamming:Vec<usize>,num:usize)->Vec<usize>{
    let mut res = vec![];
    for i in 0..hamming.len(){
        if hamming[i]==num{
            res.push(i);
        }
    }
    res
}
/// 本文を分割してハミング距離を計算する
fn get_placeid_divide(text:&str,pattern:&str)->Vec<usize>{
    let mut res = vec![];
    let text_len = text.len();
    let mut i = 0;
    while i<text.len(){
        let text_sub = &text[i..std::cmp::min(i+2*pattern.len(),text_len)];
        let placeid = get_placeid(text_sub,pattern);
        for j in placeid.iter(){
            res.push(i+*j);
        }
        i+=pattern.len();
    }
    res.dedup();
    res
}
fn get_placeid(text:&str,pattern:&str)->Vec<usize>{
    let c = calc_hamming(text, pattern);
    hamming_to_placeid(c, pattern.len())
}
/// データセットを読み込む
fn load_dataset(path:&str)->(String,String,Vec<usize>){
    let f = File::open(path).expect("file not found");
    let reader = BufReader::new(f);
    let lines = reader.lines().collect::<Result<Vec<String>, _>>().unwrap();
    let text = lines[0].clone();
    let pattern = lines[1].clone();
    let ans = if lines[2].len() == 0{vec![]}else{lines[2].split(',').map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>()};
    return (text,pattern,ans);
}

///　フォルダ内のファイル名を取得する
fn read_dir<P: AsRef<Path>>(path: P) -> io::Result<Vec<String>> {
    Ok(fs::read_dir(path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_file() {
                Some(entry.file_name().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect())
}
/// フォルダ内のデータセットを実行して速度を計算する．
fn run_by_datasets(folder:&str,func:fn(&str,&str)->Vec<usize>){
    let files = read_dir(Path::new(folder)).unwrap();
    let mut time_sec = 0;
    let mut time_nsec = 0;
    let len = files.len();
    for file in files{
        let path = format!("{}/{}",folder,file);
        let (text,pattern,ans) = load_dataset(&path);
        let (placeid,end) = measure!(func(&text,&pattern));
        assert_eq!(placeid,ans);
        time_sec += end.as_secs();
        time_nsec += end.subsec_nanos();
        if time_nsec >= 1_000_000_000{
            time_sec += 1;
            time_nsec -= 1_000_000_000;
        }
    }
    let time = time_sec as f64 + time_nsec as f64 / 1_000_000_000.0;
    println!("mean:{}/{}={}",time,len,time/(len as f64));
}
fn make_datasets(){
    make_test::make_random(5,2,100,4);
    make_test::make_random(100000,50000,10,4);
    make_test::make_substring(100000,50000,10,4);
    make_test::make_random(100000,50000,10,1);
    make_test::make_random(10000000,100,10,4);
    make_test::make_substring(10000000,100,10,4);
    make_test::make_random(10000000,100,10,1);
    make_test::make_random(100000,100,10,4);
    make_test::make_substring(100000,100,10,4);
    make_test::make_random(100000,100,10,1);
}
///追加データセットの測定
fn test_hokan(){
    let mut funcs:Vec<(fn(&str,&str)->Vec<usize>,&str)> = vec![(s_algo::brute_search,"brute")];
    funcs.push((s_algo::new_shift_and_brute,"new_shift_and_brute"));
    funcs.push((get_placeid,"FFT"));
    funcs.push((get_placeid_divide,"FFT_divide"));
    let datasets = vec!["tests/tests_random1000000_500000_4types","tests/tests_substring1000000_500000_4types","tests/tests_random1000000_500000_1types"];
    for dataset in datasets.iter(){
        for func in funcs.iter(){
            println!("dataset:{},func:{}",dataset,func.1);
           run_by_datasets(dataset,func.0);
        }
    }
}
///追加データセットの作成
fn make_hokan(){
    make_test::make_random(1000000, 500000,10, 4);
    make_test::make_substring(1000000, 500000, 10, 4);
    make_test::make_random(1000000, 500000,10, 1);
}
/// 実際の実験
fn test_by_datasets(){
    let mut funcs:Vec<(fn(&str,&str)->Vec<usize>,&str)> = vec![(s_algo::brute_search,"brute")];
    funcs.push((s_algo::new_shift_and_brute,"new_shift_and_brute"));
    funcs.push((get_placeid,"FFT"));
    funcs.push((get_placeid_divide,"FFT_divide"));
    let datasets = vec!["tests/tests_random5_2_4types","tests/tests_random100000_50000_4types","tests/tests_substring100000_50000_4types","tests/tests_random100000_50000_1types",
    "tests/tests_random100000_100_4types","tests/tests_substring100000_100_4types","tests/tests_random100000_100_1types"];
    for dataset in datasets.iter(){
        for func in funcs.iter(){
            println!("dataset:{},func:{}",dataset,func.1);
            run_by_datasets(dataset,func.0);
        }
    }
}
fn test_gosatest(){
    let mut funcs:Vec<(fn(&str,&str)->Vec<usize>,&str)> = vec![(s_algo::brute_search,"brute")];
    funcs.push((s_algo::new_shift_and_brute,"new_shift_and_brute"));
    funcs.push((get_placeid,"FFT"));
    funcs.push((get_placeid_divide,"FFT_divide"));
    let datasets = vec!["tests/tests_random1000000_500000_1types","tests/tests_random10000000_100_1types"];
    for dataset in datasets.iter(){
        for func in funcs.iter(){
            println!("dataset:{},func:{}",dataset,func.1);
            run_by_datasets(dataset,func.0);
        }
    }
}

fn main() {
    // make_datasets();
    //test_by_datasets();
    //make_gosatest();
    //make_hokan();
    test_hokan();
}
#[cfg(test)]
pub mod test{
    use super::*;
    use s_algo::*;
    #[test]
    fn test_fft_hamming(){
        let text = "AAAA";
        let pattern = "AA";
        let c = calc_hamming(text,pattern);
        let placeid = hamming_to_placeid(c, pattern.len());
        assert_eq!(placeid,vec![0,1,2]);

        let text = "B?CA";
        let pattern = "BC";
        let c = calc_hamming(text,pattern);
        let placeid = hamming_to_placeid(c, pattern.len());
        assert_eq!(placeid,vec![0,1]);
        assert_eq!(get_placeid(text, pattern),vec![0,1]);

        let text = "ABDAABDA";
        let pattern = "BDA";
        assert_eq!(get_placeid(text, pattern),vec![1,5]);

        let text = "?A?";
        let pattern = "A";
        assert_eq!(get_placeid(text, pattern),vec![0,1,2]);
    }
    #[test]
    fn test_brute(){
        let text = "AAAA";
        let pattern = "AA";
        assert_eq!(brute_search(text, pattern),vec![0,1,2]);

        let text = "BCCA";
        let pattern = "BC";
        assert_eq!(brute_search(text, pattern),vec![0]);

        let text = "ABDAABDA";
        let pattern = "BDA";
        assert_eq!(brute_search(text, pattern),vec![1,5]);

        let text = "AAA";
        let pattern = "A";
        assert_eq!(brute_search(text, pattern),vec![0,1,2]);

    }
    #[test]
    fn test_shift_and(){
        let text = "AAAA";
        let pattern = "AA";
        assert_eq!(new_shift_and(text, pattern),vec![0,1,2]);

        let text = "BCCA";
        let pattern = "BC";
        assert_eq!(new_shift_and(text, pattern),vec![0]);

        let text = "ABDAABDA";
        let pattern = "BDA";
        assert_eq!(new_shift_and(text, pattern),vec![1,5]);

        let text = "AAA";
        let pattern = "A";
        assert_eq!(new_shift_and(text, pattern),vec![0,1,2]);
    }
    fn test_by_func<F:Fn(&str,&str)->Vec<usize>>(f:F){
        let text = "AAAA";
        let pattern = "AA";
        assert_eq!(f(text, pattern),vec![0,1,2]);

        let text = "BCCA";
        let pattern = "BC";
        assert_eq!(f(text, pattern),vec![0]);

        let text = "ABDAABDA";
        let pattern = "BDA";
        assert_eq!(f(text, pattern),vec![1,5]);

        let text = "AAA";
        let pattern = "A";
        assert_eq!(f(text, pattern),vec![0,1,2]);
    }

    #[test]
    fn test_by_placeid(){
        measure!(test_by_func(get_placeid));
    }
    #[test]
    fn test_by_placeid_divide(){
        measure!(test_by_func(get_placeid_divide));
    }
    #[test]
    fn test_by_shift_and_func(){
        measure!(test_by_func(new_shift_and_brute));
    }
   #[test]
   fn test_placeid(){
    run_by_datasets("tests/tests_random1000000_500000_1types", get_placeid)
   }
   #[test]
   fn test_divide_placeid(){
    run_by_datasets("tests_AAA", get_placeid_divide)
   }
   #[test]
    fn test_by_brute(){
        run_by_datasets("tests_AAA", brute_search)
    }

    #[test]
    fn test_by_shift_and(){
        run_by_datasets("tests_AAA", new_shift_and_brute)
    }
    fn run_by_datasets(folder:&str,func:fn(&str,&str)->Vec<usize>){
        let files = read_dir(Path::new(folder)).unwrap();
        for file in files{
            let path = format!("{}/{}",folder,file);
            let (text,pattern,ans) = load_dataset(&path);
            let placeid = measure!(func(&text,&pattern));
            assert_eq!(placeid.0,ans);
        }
    }
}