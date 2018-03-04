#![feature(use_extern_macros)]

#[macro_use]
extern crate clap;
extern crate futures;
#[macro_use]
extern crate lazy_static;
extern crate protobuf;
extern crate tokio_core;

#[macro_use]
extern crate client_utils;
extern crate ekiden_core_common;
extern crate ekiden_rpc_client;

extern crate dp_credit_scoring_api;

use clap::{App, Arg};
use futures::future::Future;

use dp_credit_scoring_api::with_api;
use ekiden_rpc_client::create_client_rpc;

with_api! {
    create_client_rpc!(dpcs, dp_credit_scoring_api, api);
}

const USER: &str = "Rusty Lerner";
lazy_static! {
    static ref DATASET: dpcs::Dataset = {
        let data_output = std::process::Command::new("python2") .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../prep_data.py"))
            .args(&["--api-proto",
                  concat!(env!("CARGO_MANIFEST_DIR"), "/../../api/src/generated/api_pb2.py")
            ])
            .args(&["--max-samples", "32"])
            .output()
            .expect("Could not fetch data.");
        assert!(
            data_output.status.success(),
            "{}",
            String::from_utf8(data_output.stderr).unwrap_or("Could not generate data".to_string())
        );

        protobuf::parse_from_bytes(&data_output.stdout).expect("Unable to parse Dataset.")
    };
}

fn init<Backend>(client: &mut dpcs::Client<Backend>, _runs: usize, _threads: usize)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
    let _create_res = client
        .create({
            let mut req = dpcs::CreateRequest::new();
            req.set_requester(USER.to_string());
            req
        })
        .wait()
        .expect("error: create");

    let _train_res = client
        .train({
            let mut req = dpcs::TrainingRequest::new();
            req.set_requester(USER.to_string());
            req.set_inputs(DATASET.get_train_inputs().clone());
            req.set_targets(DATASET.get_train_targets().to_vec());
            req
        })
        .wait()
        .expect("error: train");
}

fn scenario<Backend>(client: &mut dpcs::Client<Backend>)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
    let mut _infer_res = client
        .infer({
            let mut req = dpcs::InferenceRequest::new();
            req.set_requester(USER.to_owned());
            req.set_inputs(DATASET.get_test_inputs().clone());
            req
        })
        .wait()
        .expect("error: infer");
}

fn finalize<Backend>(_client: &mut dpcs::Client<Backend>, _runs: usize, _threads: usize)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
}

fn main() {
    let results = benchmark_client!(dpcs, init, scenario, finalize);
    results.show();
}
