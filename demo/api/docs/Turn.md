# Turn

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status** | **Status** | `answered`: `answer` holds the answer. `approval`: `pending` holds the calls to decide. `stopped`: the run stopped otherwise, `detail` says why. (enum: answered, approval, stopped) | 
**answer** | Option<**String**> |  | [optional]
**detail** | Option<**String**> |  | [optional]
**pending** | Option<[**Vec<models::PendingCall>**](PendingCall.md)> |  | [optional]
**calls** | [**Vec<models::Call>**](Call.md) | The tools the assistant called in this turn, in order. | 
**requests** | **i32** | The model requests of the turn. | 
**tokens** | **i64** | Input and output tokens of the turn. | 
**trace_id** | Option<**String**> | The Langfuse trace of the turn, when tracing is on. | [optional]
**trace_url** | Option<**String**> | The trace in the Langfuse UI, when its project is known. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


