//! Standalone root-operator frontend; build/install instructions in OPERATOR.md.
//! No sudo, environment registry override, or manual-workflow grant is used.
use nf_core::canonical::{CanonicalDocument, ContentDigest, Limits};
use nf_experiments::coordinator::promotion::operator;
use std::io::Read;
fn request(path:&str)->Result<CanonicalDocument,Box<dyn std::error::Error>> {
    let mut bytes=Vec::new();
    std::fs::File::open(path)?.take(524289).read_to_end(&mut bytes)?;
    Ok(CanonicalDocument::parse(&bytes,Limits{max_bytes:524288,..Limits::default()})?)
}
fn run()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command,path] if command=="inspect"=> {
            let doc=request(path)?;
            println!("{}\n{}",String::from_utf8_lossy(doc.bytes()),doc.digest());
        },
        [command,path,expected,expires] if command=="issue"=> {
            let doc=request(path)?;
            let expected:ContentDigest=expected.parse()?;
            if doc.digest()!=expected { return Err("request changed since operator inspection".into()); }
            // Actual authorization is this separate OS principal, NOT expected.
            println!("{}",operator::issue(doc.value(),expires.parse()?)?);
        },
        [command,id] if command=="revoke"=> {
            operator::revoke(&id.parse()?)?; println!("revoked");
        },
        _=>return Err("usage: promotion-operator inspect REQUEST | issue REQUEST INSPECTED_DIGEST EXPIRES_UNIX_MS | revoke GRANT_ID".into()),
    }
    Ok(())
}
fn main() {
    if let Err(error)=run() { eprintln!("{error}"); std::process::exit(1); }
}
