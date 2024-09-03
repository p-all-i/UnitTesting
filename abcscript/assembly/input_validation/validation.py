import os, sys, json, time, datetime
from jsonschema import validate, ValidationError
from assembly.input_validation.schema import modelInfo_schema, cameraInfo_schema, trackerInfo_schema


class validateInput:

    def validate(schema, json):
        try:
            validate(instance=json, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        
    def validate_main(inputdata, loggerObj):
        keys_lt = ["modelsInfo", "cameraInfo"]
        schema_lt = [modelInfo_schema, cameraInfo_schema]
        pass_lt = []

        for key, schema in zip(keys_lt, schema_lt):
            input_json = inputdata[key]
            validation_res, exeption_reason = validateInput.validate(schema=schema, json=input_json)
            if validation_res:
                pass_lt.append(True)
                print(f"[INFO] {datetime.datetime.now()} Input validation completed for {key}")
                loggerObj.logger.info(f"Input validation completed for {key}")
            else:
                pass_lt.append(False)
                print(f"[INFO] {datetime.datetime.now()} Input validation failed for {key}")
                print(f"[INFO] {datetime.datetime.now()} Validation failure reason {exeption_reason}")
                loggerObj.logger.info(f"Input validation failed for {key}")
                loggerObj.logger.info(f"Validation failure reason {exeption_reason}")
        return all(pass_lt)

