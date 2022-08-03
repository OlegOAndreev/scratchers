#[link(name = "div", kind = "static")]
extern "C" {
    pub fn my_div(x: u32, y: u32) -> u32;
}

fn main() {
    unsafe {
        println!("{}/{} = {}", 1, 0, my_div(1, 0));
        println!("{}/{} = {}", 2, 0, my_div(2, 0));
        println!("{}/{} = {}", 1, 1, my_div(1, 1));
        println!("{}/{} = {}", 2, 1, my_div(2, 1));
        println!("{}/{} = {}", 3, 1, my_div(3, 1));
        println!("{}/{} = {}", 1, 2, my_div(1, 2));
        println!("{}/{} = {}", 2, 2, my_div(2, 2));
        println!("{}/{} = {}", 3, 2, my_div(3, 2));
        println!("{}/{} = {}", 4, 2, my_div(4, 2));
        println!("{}/{} = {}", 5, 2, my_div(5, 2));
        println!("{}/{} = {}", 1, 3, my_div(1, 3));
        println!("{}/{} = {}", 2, 3, my_div(2, 3));
        println!("{}/{} = {}", 3, 3, my_div(3, 3));
        println!("{}/{} = {}", 4, 3, my_div(4, 3));
        println!("{}/{} = {}", 5, 3, my_div(5, 3));
        println!("{}/{} = {}", 6, 3, my_div(6, 3));
        println!("{}/{} = {}", 7, 3, my_div(7, 3));
        println!("{}/{} = {} ({})", 0x80000000u32, 1, my_div(0x80000000, 1), 0x80000000u32 / 1);
        println!("{}/{} = {} ({})", 0x80000000u32, 2, my_div(0x80000000, 2), 0x80000000u32 / 2);
        println!("{}/{} = {} ({})", 0x80000000u32, 3, my_div(0x80000000, 3), 0x80000000u32 / 3);

        let margin = 1000u32;
        let mid = 0x80000000u32;
        for i in 0..margin {
            for j in 1..margin {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (u32::MAX - margin)..u32::MAX {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (mid - margin)..(mid + margin) {
                assert_eq!(i / j, my_div(i, j));
            }
        }
        for i in (u32::MAX - margin)..u32::MAX {
            for j in 1..margin {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (u32::MAX - margin)..u32::MAX {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (mid - margin)..(mid + margin) {
                assert_eq!(i / j, my_div(i, j));
            }
        }
        for i in (mid - margin)..(mid + margin) {
            for j in 1..margin {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (u32::MAX - margin)..u32::MAX {
                assert_eq!(i / j, my_div(i, j));
            }
            for j in (mid - margin)..(mid + margin) {
                assert_eq!(i / j, my_div(i, j));
            }
        }
    }
}
