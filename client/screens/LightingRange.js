import { useFocusEffect } from "@react-navigation/native";
import React, { useCallback, useState } from "react";
import * as Strings from '../constants/string';
import * as Errors from '../constants/error';
import * as APIs from '../constants/api';
import * as http from "../utils/http";
import RangeSelect from "../components/RangeSelect";
import Loading from "../components/Loading";
import { MyTheme } from "../constants/theme";

const LightingRange = () => {
    const [minValue, setMinValue] = useState("30");
    const [maxValue, setMaxValue] = useState("130");
    const [loading, setLoading] = useState(true);

    useFocusEffect(
        useCallback(() => {
            http.get('server', APIs.LIGHT_INTENSITY_RANGE)
                .then((data) => {
                    setMinValue(data.minLightEnergy.toString());
                    setMaxValue(data.maxLightEnergy.toString());
                    setLoading(false);

                    console.log("Got min: " + data.minLightEnergy);
                    console.log("Got max: " + data.maxLightEnergy);
                })
        }, [])
    );

    function handleSave() {
        console.log(minValue);
        console.log(maxValue);

        http.request(
            'server',
            'PUT', 
            APIs.LIGHT_INTENSITY_RANGE,
            {
                minLightEnergy: parseInt(minValue),
                maxLightEnergy: parseInt(maxValue)
            }
        );
    }

    return (
        loading ? <Loading color={MyTheme.yellow} /> :
        <RangeSelect 
            minValue={minValue}
            maxValue={maxValue}
            setMinValue={setMinValue}
            setMaxValue={setMaxValue}
            onSave={handleSave}
            minPossible={0}
            maxPossible={Infinity}
            unit={Strings.LIGHT_INTENSITY_UNIT}
        />
    );
}

export default LightingRange;