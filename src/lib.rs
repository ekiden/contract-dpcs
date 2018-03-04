#![feature(use_extern_macros)]

extern crate ndarray;
extern crate protobuf;
extern crate serde;
extern crate serde_cbor;
#[macro_use]
extern crate serde_derive;

#[cfg(not(target_env = "sgx"))]
extern crate rand as rand;
#[cfg(target_env = "sgx")]
extern crate sgx_rand as rand;

extern crate ekiden_core_common;
extern crate ekiden_core_trusted;

extern crate dp_credit_scoring_api;

mod contract;
mod dpml;

use ndarray::Array;

use ekiden_core_common::{Error, Result};
use ekiden_core_common::contract::{Address, Contract};
use ekiden_core_trusted::db::Db;
use ekiden_core_trusted::rpc::create_enclave_rpc;

use dp_credit_scoring_api::*;

use contract::Learner;

// Create enclave RPC handlers.
with_api! {
    create_enclave_rpc!(api);
}

macro_rules! unpack {
    ($req:ident) => {
        {
            let state = Db::instance().get("state")?;
            let learner = Learner::from_state(&state);
            if !Address::from($req.get_requester().to_string()).eq(learner.get_owner()?) {
                return Err(Error::new("Insufficient permissions."));
            }
            learner
        }
    }
}

fn create(req: &CreateRequest) -> Result<CreateResponse> {
    let learner = Learner::new(Address::from(req.get_requester().to_string()));
    Db::instance().set("state", learner.get_state())?;
    Ok(CreateResponse::new())
}

fn train(req: &TrainingRequest) -> Result<TrainingResponse> {
    let mut learner = unpack!(req);

    let inputs = req.get_inputs();
    let xs = Array::from_shape_vec(
        (inputs.get_rows() as usize, inputs.get_cols() as usize),
        inputs.get_data().iter().map(|&v| v as f64).collect(),
    ).unwrap();
    let targets = req.get_targets();
    let ys = Array::from_shape_vec(
        (targets.len() as usize, 1),
        targets.iter().map(|&v| v as f64).collect(),
    ).unwrap();
    learner.train(&xs, &ys)?;

    Db::instance().set("state", learner.get_state())?;

    Ok(TrainingResponse::new())
}

fn infer(req: &InferenceRequest) -> Result<InferenceResponse> {
    let learner = unpack!(req);

    let inputs = req.get_inputs();
    let xs = Array::from_shape_vec(
        (inputs.get_rows() as usize, inputs.get_cols() as usize),
        inputs.get_data().iter().map(|&v| v as f64).collect(),
    ).unwrap();
    let preds = learner.infer(&xs)?;

    let mut response = InferenceResponse::new();
    response.set_predictions(preds.iter().map(|&v| v as f32).collect::<Vec<f32>>());
    Ok(response)
}
