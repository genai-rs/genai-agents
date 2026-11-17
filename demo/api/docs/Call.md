# Call

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **String** |  | 
**arguments** | **String** | The JSON arguments the model gave. | 
**output** | Option<**String**> | The tool's answer, shortened to 4000 characters; absent while the call waits for approval. | [optional]
**ok** | Option<**bool**> | Whether the tool succeeded; absent while the call waits for approval. | [optional]
**duration_ms** | Option<**i64**> | From the call to its answer, in milliseconds; absent for a call approved in an earlier turn. | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


