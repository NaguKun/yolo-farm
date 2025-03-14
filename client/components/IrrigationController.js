import React, { useCallback, useState } from "react";
import * as Strings from "../constants/string";
import * as Headers from "../constants/header";
import * as Modes from "../constants/mode";
import * as APIs from "../constants/api";
import { StyleSheet, View } from "react-native";
import { faDroplet, faGear, faWarning } from "@fortawesome/free-solid-svg-icons";
import { faCalendar } from "@fortawesome/free-regular-svg-icons";
import { AnimatedCircularProgress } from 'react-native-circular-progress';
import DeviceToggler from "./DeviceToggler";
import SettingItem from "./SettingItem";
import { Text } from "react-native-paper";
import { FontAwesomeIcon } from "@fortawesome/react-native-fontawesome";
import { MyTheme } from "../constants/theme";
import * as mqtt from "../utils/mqtt";
import * as http from "../utils/http";
import { useFocusEffect } from "@react-navigation/native";
import Loading from "./Loading";
import ControllerLayout from "../layouts/ControllerLayout";

const IrrigationController = () => {

    const [soilMoisture, setSoilMoisture] = useState(0);
    const [pumping, setPumping] = useState(false);
    const [minValue, setMinValue] = useState(null);
    const [maxValue, setMaxValue] = useState(null);
    const [mode, setMode] = useState(Modes.MANUAL);
    const [loading, setLoading] = useState(true);
    const warning = soilMoisture < minValue || soilMoisture > maxValue;

    useFocusEffect(
        useCallback(() => {

            const client = mqtt.init();
            mqtt.connect(client, [APIs.SOIL_MOISTURE_FEED, APIs.PUMP_FEED]);
            client.onMessageArrived = (message) => {
                console.log("Topic: " + message.destinationName);
                console.log("Message: " + message.payloadString);
        
                topic = message.destinationName;
                data = message.payloadString;
                if (topic === APIs.PUMP) {
                    setPumping(data == "1")
                } else if (topic === APIs.SOIL_MOISTURE) {
                    setSoilMoisture(parseInt(data));
                }
            }

            Promise.all([
                http.get('adafruit', APIs.PUMP_FEED),
                http.get('adafruit', APIs.SOIL_MOISTURE_FEED),
                http.get('server', APIs.PUMP_MODE),
                http.get('server', APIs.SOIL_MOISTURE_RANGE)
            ]).then(([
                pumpData, 
                soilMoisureData, 
                pumpModeData,
                rangeData
            ]) => {
                setPumping(pumpData.last_value);
                setSoilMoisture(parseInt(soilMoisureData.last_value));
                setMode(pumpModeData.mode);
                setMinValue(rangeData.minMoisture);
                setMaxValue(rangeData.maxMoisture);
                setLoading(false);
            })

            // http.get('adafruit', APIs.PUMP_FEED)
            // .then((data) => {
            //     setPumping(data.last_value);

            //     console.log("Got pump: " + data.last_value);
            // });

            // http.get('adafruit', APIs.SOIL_MOISTURE_FEED)
            //     .then((data) => {
            //         setSoilMoisture(parseInt(data.last_value));

            //         console.log("Got moisture: " + data.last_value);
            //     });

            // return () => {
            //     mqtt.disconnect(client);
            // }
        }, [])
    );
    
    // useFocusEffect(
    //     useCallback(() => {

    //         http.get('server', APIs.PUMP_MODE)
    //             .then((data) => {
    //                 setMode(data.mode)

    //                 console.log("Got mode: " + mode);
    //             });

    //         http.get('server', APIs.SOIL_MOISTURE_RANGE)
    //             .then((data) => {
    //                 setMinValue(data.minMoisture);
    //                 setMaxValue(data.maxMoisture);

    //                 console.log("Got min: " + data.minMoisture);
    //                 console.log("Got max: " + data.maxMoisture);
    //             })

    //         return () => {
    //             // mqtt.disconnect(client);
    //         }
    //     }, [])
    // );

    function onTogglePump() {
        console.log("Toggled!");

        http.request(
            'adafruit', 
            'POST', 
            APIs.PUMP_DATA,
            {
                value: pumping ? '0' : '1'
            }
        );

        setPumping(!pumping);
    }

    const devices = (
        <DeviceToggler 
            deviceName={Strings.WATER_PUMP}
            icon={faDroplet}
            enabled={pumping} 
            onSwitch={onTogglePump}
            disabled={mode !== Modes.MANUAL}
            color={MyTheme.blue}
        />
    );

    const meters = (
        <View style={{
            flex: 1,
            alignItems: 'center',
            gap: 20
        }}>
            <View style={{flex: 1, flexDirection: 'row', gap: 10 }}>
                { warning ? <FontAwesomeIcon icon={faWarning} color={MyTheme.red} size={32}/> : null}
                <Text style={{ fontSize: 24, color: warning ? MyTheme.red : 'black'}}>{Strings.SOIL_MOISTURE}</Text>
            </View>
            <AnimatedCircularProgress
                size={200}
                width={20}
                fill={soilMoisture}
                rotation={0}
                tintColor={ warning ? MyTheme.red : MyTheme.blue }
                // onAnimationComplete={() => console.log('onAnimationComplete')}
                backgroundColor="#CCCCCC">
                    {
                        () => (
                            <>
                                <Text style={{ fontSize: 50 }}>
                                    {soilMoisture}
                                </Text>
                                <Text style={{ fontSize: 25 }}>%</Text>
                            </>
                        )
                    } 
            </AnimatedCircularProgress>
        </View>                    
    );

    const settings = (
        <View style={{ 
            flex: 1, 
            flexDirection: 'row', 
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 10
        }}>
            <SettingItem 
                title={Strings.MODE} 
                content={Modes.modeTitles[mode]}
                icon={faGear} 
                target={Headers.PUMP_MODE} 
                primColor={MyTheme.darkblue}
                bgColor={MyTheme.lightblue}
            />
            <SettingItem 
                title={Strings.ALLOWED_RANGE} 
                content={`${minValue} - ${maxValue}%`}
                icon={faWarning}
                target={Headers.SOIL_MOISTURE_RANGE}
                primColor={MyTheme.darkblue}
                bgColor={MyTheme.lightblue}
            />
        </View>
    );

    return (
        loading ? <Loading color={MyTheme.blue} /> :
        <ControllerLayout 
            devices={devices}
            meters={meters}
            settings={settings}
        />
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        width: '100%',
        // backgroundColor: '#C3E6FF'
    },
    devices: {
        flex: 1,
        padding: 20,
        gap: 20,
        // backgroundColor: 'orange',
        borderRadius: 30
    },
    metrics: {
        flex: 3,
        padding: 20,
        gap: 20,
        // backgroundColor: 'skyblue',
        borderRadius: 30
    },
    settings: {
        flex: 1.5,
        padding: 20,
        gap: 20,
        // backgroundColor: 'lightgreen',
        borderRadius: 30
    },
    meter: {
        flex: 1,
        alignItems: 'center',
        gap: 20
    },

    header: {
        fontSize: 30
    }
})

export default IrrigationController;