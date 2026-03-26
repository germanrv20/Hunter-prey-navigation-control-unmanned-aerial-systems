// MESSAGE CONTROL_STATUS support class

#pragma once

namespace mavlink {
namespace development {
namespace msg {

/**
 * @brief CONTROL_STATUS message
 *
 * Information about GCS in control of this MAV. This should be broadcast at low rate (nominally 1 Hz) and emitted when ownership or takeover status change. Control over MAV is requested using MAV_CMD_REQUEST_OPERATOR_CONTROL.
 */
struct CONTROL_STATUS : mavlink::Message {
    static constexpr msgid_t MSG_ID = 512;
    static constexpr size_t LENGTH = 2;
    static constexpr size_t MIN_LENGTH = 2;
    static constexpr uint8_t CRC_EXTRA = 184;
    static constexpr auto NAME = "CONTROL_STATUS";


    uint8_t sysid_in_control; /*<  System ID of GCS MAVLink component in control (0: no GCS in control). */
    uint8_t flags; /*<  Control status. For example, whether takeover is allowed, and whether this message instance defines the default controlling GCS for the whole system. */


    inline std::string get_name(void) const override
    {
            return NAME;
    }

    inline Info get_message_info(void) const override
    {
            return { MSG_ID, LENGTH, MIN_LENGTH, CRC_EXTRA };
    }

    inline std::string to_yaml(void) const override
    {
        std::stringstream ss;

        ss << NAME << ":" << std::endl;
        ss << "  sysid_in_control: " << +sysid_in_control << std::endl;
        ss << "  flags: " << +flags << std::endl;

        return ss.str();
    }

    inline void serialize(mavlink::MsgMap &map) const override
    {
        map.reset(MSG_ID, LENGTH);

        map << sysid_in_control;              // offset: 0
        map << flags;                         // offset: 1
    }

    inline void deserialize(mavlink::MsgMap &map) override
    {
        map >> sysid_in_control;              // offset: 0
        map >> flags;                         // offset: 1
    }
};

} // namespace msg
} // namespace development
} // namespace mavlink
