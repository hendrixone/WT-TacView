from construct import Enum, Int16ul, Const, Struct, Int32ub, Int32ul, Bytes, this, Int24ul, Seek, Switch, If, \
    IfThenElse, Pass, Probe, PaddedString

simple_blk = "blk" / Struct(
    "magic" / Const(b"\x3C"),
    # "magic" / Const(b"\x00BBF"),
    "unknown_0" / Int32ub,
    "blk_body_size" / Int32ul,
    # maybe find more suitable format?
    "blk_body" / Bytes(this.blk_body_size),
)

wrpl_file = "wrpl" / Struct(
    "magic" / Const(b"\xe5\xac\x00\x10"),
    "version" / Int32ul,
    'map_file' / PaddedString(128, "utf-8"),
    'mission_file' / PaddedString(260, "utf-8"),
    'mission_type' / PaddedString(128, "utf-8"),
    'time' / PaddedString(128, "utf-8"),
    'visibility' / PaddedString(32, "utf-8"),
    Seek(0x4C4),
    "m_set" / simple_blk,
)



with open("P-38 Rurh.wrpl", 'rb') as file:
    file = file.read()
    parsed_data = wrpl_file.parse(file)
    print(parsed_data)
