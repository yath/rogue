use byteorder::{LittleEndian, ReadBytesExt};
use crc::{Crc, CRC_32_ISO_HDLC};
use log::{debug, info, warn};
use midir::{MidiIO, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use regex::Regex;
use snafu::prelude::*;
use std::{
    error,
    fmt::{self, Display},
    fs::File,
    io::{self, Cursor, ErrorKind, Read},
    path::Path,
    str::FromStr,
    sync::mpsc,
    time::Duration,
};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString, FromRepr, IntoStaticStr};
use zip::ZipArchive;

static CLIENT_IDENTIFICATION: &str = "rogue";
static MIDI_CONNECTION_IDENTIFICATION: &str = "rogue connection";
const READ_TIMEOUT: Duration = Duration::from_secs(5);

type Result<T> = std::result::Result<T, Error>;

fn undo_7bitize(input: &[u8]) -> Vec<u8> {
    let mut n_chunks = input.len() / 8;
    if input.len() % 8 != 0 {
        n_chunks += 1;
    }

    let mut ret: Vec<u8> = Vec::with_capacity(n_chunks * 7);
    for chunk in 0..n_chunks {
        let msbs = input[chunk * 8];
        for byte in 1..8 {
            if let Some(val) = input.get(chunk * 8 + byte) {
                let shift = byte - 1;
                let msb = (msbs >> shift) & 1;
                ret.push(msb << 7 | val);
            } else {
                break;
            }
        }
    }

    ret
}

fn do_7bitize(input: &[u8]) -> Vec<u8> {
    let mut n_chunks = input.len() / 7;
    if input.len() % 7 != 0 {
        n_chunks += 1;
    }

    let mut ret: Vec<u8> = Vec::with_capacity(n_chunks * 8);
    for chunk in 0..n_chunks {
        ret.push(0); // reserve for msbs
        let mut msbs = 0u8;
        for byte in 0..7 {
            if let Some(val) = input.get(chunk * 7 + byte) {
                let msb = val >> 7;
                msbs |= msb << byte;
                ret.push(val & 0x7f);
            } else {
                break;
            }
        }
        ret[chunk * 8] = msbs;
    }
    ret
}

fn remaining_slice<'a>(c: &Cursor<&'a [u8]>) -> &'a [u8] {
    let buf = c.get_ref();
    &buf[c.position() as usize..buf.len()]
}

fn string_from_sz(sz: &[u8]) -> String {
    let end = sz.iter().position(|c| *c == 0).unwrap_or(sz.len());
    String::from_utf8_lossy(&sz[0..end]).into()
}

fn string_to_padded_sz(s: String, l: usize) -> Vec<u8> {
    let mut ret = s.into_bytes();
    if ret.len() > l {
        warn!("Truncating string to {} bytes", l);
    }
    ret.resize(l, 0);

    ret
}

fn check_empty(mut c: io::Cursor<&[u8]>) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    c.read_to_end(&mut buf).context(DeviceReadIOContext {})?;

    if buf.is_empty() {
        Ok(())
    } else {
        DeviceResponseParseContext {
            message: "buffer not empty",
            offset: None,
            buf,
        }
        .fail()
    }
}

fn get_matching_device<T: MidiIO>(io: &T) -> Result<T::Port> {
    let all_ports = io.ports();
    let mut matching_ports = all_ports
        .into_iter()
        .filter(|port| match io.port_name(port) {
            // FIXME match port properly
            Ok(p) => p.starts_with("NTS-1"),
            Err(e) => {
                eprintln!("Error looking up port: {}", e);
                false
            }
        })
        .collect::<Vec<T::Port>>();

    match matching_ports.len() {
        0 => DeviceDiscoveryContext {
            message: "no matching device found",
        }
        .fail(),
        1 => Ok(matching_ports.swap_remove(0)),
        _ => DeviceDiscoveryContext {
            message: "more than one matching device found",
        }
        .fail(),
    }
}

pub fn get_logue_device() -> Result<Device> {
    let mut midi_input =
        MidiInput::new(CLIENT_IDENTIFICATION).map_err(|e| Error::DeviceCommunicationError {
            source: Box::new(e),
        })?;
    midi_input.ignore(midir::Ignore::Time);
    let input_port =
        get_matching_device(&midi_input).map_err(|e| Error::DeviceCommunicationError {
            source: Box::new(e),
        })?;
    let (send, recv) = mpsc::channel::<Vec<u8>>();
    let input_conn = midi_input
        .connect(
            &input_port,
            MIDI_CONNECTION_IDENTIFICATION,
            move |ts, message, _| {
                debug!("Received message at {}: {:x?}", ts, message);
                send.send(Vec::from(message)).unwrap();
            },
            (),
        )
        .map_err(|e| Error::DeviceCommunicationError {
            source: Box::new(e),
        })?;

    let midi_output =
        MidiOutput::new(CLIENT_IDENTIFICATION).map_err(|e| Error::DeviceCommunicationError {
            source: Box::new(e),
        })?;
    let output_port = get_matching_device(&midi_output)?;
    let output_conn = midi_output
        .connect(&output_port, MIDI_CONNECTION_IDENTIFICATION)
        .map_err(|e| Error::DeviceCommunicationError {
            source: Box::new(e),
        })?;

    Ok(Device {
        _input: input_conn,
        input_channel: recv,
        output: output_conn,
    })
}

#[derive(Debug, Snafu)]
#[snafu(context(suffix(Context)))]
pub enum Error {
    #[snafu(display(
        "Could not parse response from device at offset 0x{offset:x?}: {message}. Data: {buf:x?}"
    ))]
    DeviceResponseParseError {
        message: String,
        offset: Option<usize>,
        buf: Vec<u8>,
    },

    #[snafu(display("Unable to read from device: {source}"))]
    DeviceReadIOError { source: io::Error },

    #[snafu(display("Timeout waiting for response from device: {source}"))]
    DeviceReadTimeout { source: mpsc::RecvTimeoutError },

    #[snafu(display("Unable to communicate with device: {source}"))]
    DeviceCommunicationError {
        #[snafu(source(from(midir::SendError, Box::new)))]
        source: Box<dyn error::Error>,
    },

    #[snafu(display("Could not parse manifest: {message}"))]
    InvalidManifestError { message: String },

    #[snafu(display("Could not deserialize manifest: {source}"))]
    ManifestSerdeError { source: serde_json::Error },

    #[snafu(display("Could not read from manifest's {filename}: {source}"))]
    ManifestFileIOError { filename: String, source: io::Error },

    #[snafu(display("Could not read unit file: {source}"))]
    ManifestIOError { source: io::Error },

    #[snafu(display("Could not unpack unit file: {source}"))]
    ManifestZIPError { source: zip::result::ZipError },

    #[snafu(display("Payload error: {message}"))]
    PayloadError { message: String },

    #[snafu(display("Unable to discover device: {message}"))]
    DeviceDiscoveryError { message: String },
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceIdentity {
    pub channel: u8,
    pub manufacturer: u8,
    pub family_id: u16,
    pub member_id: u16,
    pub major_ver: u16,
    pub minor_ver: u16,
}

impl DeviceIdentity {
    fn from_repr(msg: &[u8]) -> Result<DeviceIdentity> {
        if msg.len() != 15 {
            return DeviceResponseParseContext {
                message: "invalid length",
                offset: None,
                buf: msg,
            }
            .fail();
        }

        let msg = msg
            .strip_prefix(&[0xf0, 0x7e])
            .context(DeviceResponseParseContext {
                message: "invalid prefix, want 0xf0 0x7e",
                offset: Some(0),
                buf: msg,
            })?
            .strip_suffix(&[0xf7])
            .context(DeviceResponseParseContext {
                message: "invalid suffix, want 0xf7",
                offset: Some(msg.len() - 1),
                buf: msg,
            })?;

        let mut c = Cursor::new(msg);
        let channel = read_u8(&mut c)?;
        if read_u16_le(&mut c)? != 0x0206 {
            return DeviceResponseParseContext {
                message: "invalid message type, want identity reply (0x06 0x02)",
                offset: Some(2),
                buf: msg,
            }
            .fail();
        }
        let manufacturer = read_u8(&mut c)?;
        if manufacturer != 0x42 {
            warn!("Manufacturer ID 0x{:x} != KORG (0x42)", manufacturer);
        }
        let (family_id, member_id) = (read_u16_le(&mut c)?, read_u16_le(&mut c)?);
        let (major_ver, minor_ver) = (read_u16_le(&mut c)?, read_u16_le(&mut c)?);

        if let Some(dev) = Platform::from_device_family(family_id) {
            info!("Device recognized as {}", dev);
        } else {
            warn!(
                "Unknown device family 0x{:x}, but continuing anyway",
                family_id
            );
        }

        let s = DeviceIdentity {
            channel,
            manufacturer,
            family_id,
            member_id,
            major_ver,
            minor_ver,
        };

        Ok(s)
    }

    pub fn platform(&self) -> Option<Platform> {
        Platform::from_device_family(self.family_id)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Version(pub u8, pub u8, pub u8);

impl Version {
    fn from_str(s: &str) -> Option<Self> {
        let regex = Regex::new(r"^(\d{1,2})\.(\d{1,2})-(\d{1,2})$").unwrap();
        let caps = match regex.captures(s) {
            None => return None,
            Some(caps) => caps,
        };

        let get_u8 = |i| match caps.get(i) {
            None => None,
            Some(cap) => match cap.as_str().parse::<u8>() {
                Err(_) => None,
                Ok(val) => Some(val),
            },
        };

        Some(Version(get_u8(1)?, get_u8(2)?, get_u8(3)?))
    }

    fn as_repr(&self) -> [u8; 4] {
        [self.2, self.1, self.0, 0]
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{:02}-{}", self.0, self.1, self.2)
    }
}

#[repr(u8)]
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Display, FromRepr, EnumString,
)]
pub enum Platform {
    #[strum(to_string = "prologue")]
    Prologue = 0x1,
    #[strum(to_string = "minilogue xd", serialize = "minilogue-xd")]
    MinilogueXD = 0x2,
    #[strum(to_string = "nutekt digital", serialize = "nutekt-digital")]
    NutektDigital = 0x3,
    #[strum(disabled)]
    Unknown = 0xff,
}

impl Platform {
    fn from_device_family(f: u16) -> Option<Self> {
        match f {
            0x14b => Some(Self::Prologue),
            0x151 | 0x12c => Some(Self::MinilogueXD), // disassembly says 0x151, doc says 0x12c
            0x157 => Some(Self::NutektDigital),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserAPIVersion {
    pub platform: Platform,
    pub version: Version,
}

// Reads an u8 from c with io::error converted to Error::DeviceReadIOError.
fn read_u8(c: &mut io::Cursor<&[u8]>) -> Result<u8> {
    c.read_u8().context(DeviceReadIOContext {})
}

// Reads a little-endian u16 from c with io::error converted to Error::DeviceReadIOError.
fn read_u16_le(c: &mut io::Cursor<&[u8]>) -> Result<u16> {
    c.read_u16::<LittleEndian>().context(DeviceReadIOContext {})
}
// Reads a little-endian u32 from c with io::error converted to Error::DeviceReadIOError.
fn read_u32_le(c: &mut io::Cursor<&[u8]>) -> Result<u32> {
    c.read_u32::<LittleEndian>().context(DeviceReadIOContext {})
}

impl UserAPIVersion {
    fn from_repr(msg: &[u8]) -> Result<Self> {
        let mut c = io::Cursor::new(msg);
        let ret = UserAPIVersion {
            platform: Platform::from_repr(read_u8(&mut c)?).unwrap_or(Platform::Unknown),
            version: Version(read_u8(&mut c)?, read_u8(&mut c)?, read_u8(&mut c)?),
        };
        check_empty(c)?;
        Ok(ret)
    }
}

impl Display for UserAPIVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Platform {}, API {}", self.platform, self.version)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserSlotStatusUnknownFields(pub u8, pub u8, pub u8, pub Vec<u8>);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserSlotStatus {
    pub api_version: UserAPIVersion,
    pub module: UserModuleType,
    pub slot: u8,
    pub developer_id: u32,
    pub program_id: u32,
    pub program_version: Version,
    pub program_name: String,
    pub num_params: u8,

    // Additional / Unknown fields
    pub module2: UserModuleType,
    pub unknown: UserSlotStatusUnknownFields,
}

impl UserSlotStatus {
    fn from_repr(msg: &[u8]) -> Result<Option<Self>> {
        let mut c = Cursor::new(msg);
        let (module_type, slot) = (read_u8(&mut c)?, read_u8(&mut c)?); // [0,2)
        let unk1: u8 = match read_u8(&mut c) {
            Ok(b) => b,
            Err(Error::DeviceReadIOError { source })
                if source.kind() == ErrorKind::UnexpectedEof =>
            {
                return Ok(None)
            }
            Err(e) => return Err(e),
        }; // [2,3)

        // Remaining data is "7-bit-ized".
        let msg = undo_7bitize(remaining_slice(&c));
        let mut c = Cursor::new(msg.as_ref());

        let (module_type2, platform_id) = (read_u8(&mut c)?, read_u8(&mut c)?); // [0,2)
        let (api_patch, api_minor, api_major, unk2) = (
            read_u8(&mut c)?,
            read_u8(&mut c)?,
            read_u8(&mut c)?,
            read_u8(&mut c)?,
        ); // [2,6)
        let (did, uid) = (read_u32_le(&mut c)?, read_u32_le(&mut c)?); // [6,14)
        let (mod_patch, mod_minor, mod_major, unk3) = (
            read_u8(&mut c)?,
            read_u8(&mut c)?,
            read_u8(&mut c)?,
            read_u8(&mut c)?,
        ); // [14,18)

        let mut name0 = [0; 14];
        c.read_exact(&mut name0).context(DeviceReadIOContext)?; // [18..32)
        let name = string_from_sz(&name0);

        let nparams = read_u8(&mut c)?;
        let unk4 = remaining_slice(&c).to_vec();

        let ret = UserSlotStatus {
            api_version: UserAPIVersion {
                platform: Platform::from_repr(platform_id).unwrap_or(Platform::Unknown),
                version: Version(api_major, api_minor, api_patch),
            },
            module: UserModuleType::from_repr(module_type).unwrap_or(UserModuleType::Unknown),
            slot,
            developer_id: did,
            program_id: uid,
            program_version: Version(mod_major, mod_minor, mod_patch),
            program_name: name,
            num_params: nparams,

            module2: UserModuleType::from_repr(module_type2).unwrap_or(UserModuleType::Unknown),
            unknown: UserSlotStatusUnknownFields(unk1, unk2, unk3, unk4),
        };

        Ok(Some(ret))
    }
}

impl Display for UserSlotStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\"{name}\" {version} {api_version} did:{did:08X} uid:{uid:08X}",
            name = self.program_name,
            version = self.program_version,
            api_version = self.api_version,
            did = self.developer_id,
            uid = self.program_id
        )
    }
}

mod json_manifest {
    use serde::Deserialize;

    #[derive(Debug, Eq, PartialEq, Deserialize)]
    pub struct Param(pub String, pub u8, pub u8, pub String);

    #[derive(Debug, Eq, PartialEq, Deserialize)]
    pub struct Header {
        pub platform: String,
        pub module: String,
        pub api: String,
        pub dev_id: u32,
        pub prg_id: u32,
        pub version: String,
        pub name: String,
        pub num_param: u8,
        pub params: Option<Vec<Param>>,
    }

    #[derive(Debug, Eq, PartialEq, Deserialize)]
    pub struct Manifest {
        pub header: Header,
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModuleParam {
    pub name: String,
    pub min: u8,
    pub max: u8,
    pub is_percentage: bool,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModuleManifest {
    pub api: UserAPIVersion,
    pub module: UserModuleType,
    pub developer_id: u32,
    pub program_id: u32,
    pub program_version: Version,
    pub name: String,
    pub num_params: u8,
    pub params: Vec<ModuleParam>,
}

impl ModuleManifest {
    pub fn from_reader<R: Read>(r: R) -> Result<Self> {
        let j: json_manifest::Manifest =
            serde_json::from_reader(r).context(ManifestSerdeContext)?;
        ModuleManifest::try_from(j)
    }

    fn into_repr(self) -> Result<Vec<u8>> {
        let mut ret = vec![self.module as u8, self.api.platform as u8]; // [0,2)
        ret.extend(self.api.version.as_repr()); // [2..6)
        ret.extend(self.developer_id.to_le_bytes()); // [6..10)
        ret.extend(self.program_id.to_le_bytes()); // [10..14)
        ret.extend(self.program_version.as_repr()); // [14..18)
        ret.extend(string_to_padded_sz(self.name, 14)); // [18..32)
        ret.extend((self.num_params as u32).to_le_bytes()); // [32..36)
        if self.num_params as usize != self.params.len() {
            return Err(Error::InvalidManifestError {
                message: format!(
                    "num_param ({}) does not match number of params ({})",
                    self.num_params,
                    self.params.len()
                ),
            });
        }
        for p in self.params.into_iter() {
            let unit: u8 = if p.is_percentage { 0 } else { 2 };
            ret.extend([p.min, p.max, unit]);
            ret.extend(string_to_padded_sz(p.name, 13));
        } // [36..36+16*num_param)

        Ok(ret)
    }
}

impl TryFrom<json_manifest::Manifest> for ModuleManifest {
    type Error = Error;

    fn try_from(j: json_manifest::Manifest) -> Result<Self> {
        Ok(Self {
            api: UserAPIVersion {
                platform: match Platform::from_str(&j.header.platform) {
                    Err(e) => {
                        return InvalidManifestContext {
                            message: format!("Unknown platform {} ({})", &j.header.platform, e),
                        }
                        .fail()
                    }
                    Ok(p) => p,
                },
                version: match Version::from_str(&j.header.api) {
                    None => {
                        return InvalidManifestContext {
                            message: format!("Unable to parse API version {}", &j.header.api),
                        }
                        .fail()
                    }
                    Some(v) => v,
                },
            },
            module: UserModuleType::from_str(&j.header.module).map_err(|err| {
                Error::InvalidManifestError {
                    message: format!("can't parse module {}: {}", j.header.module, err),
                }
            })?,
            developer_id: j.header.dev_id,
            program_id: j.header.prg_id,
            program_version: match Version::from_str(j.header.version.as_ref()) {
                None => {
                    return InvalidManifestContext {
                        message: format!("can't parse program version {}", j.header.version),
                    }
                    .fail()
                }
                Some(v) => v,
            },
            name: j.header.name,
            num_params: j.header.num_param,
            params: j
                .header
                .params
                .unwrap_or_default()
                .into_iter()
                .map(|p| ModuleParam {
                    name: p.0,
                    min: p.1,
                    max: p.2,
                    is_percentage: match p.3.as_ref() {
                        "%" => true,
                        "" => false,
                        _ => {
                            warn!("Unknown unit '{}'", p.3);
                            false
                        }
                    },
                })
                .collect(),
        })
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModulePackage {
    pub manifest: ModuleManifest,
    pub payload: Vec<u8>,
}

impl ModulePackage {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut zip = ZipArchive::new(File::open(path).context(ManifestIOContext)?)
            .context(ManifestZIPContext)?;
        let mut payload: Option<Vec<u8>> = None;
        let mut manifest: Option<ModuleManifest> = None;
        for i in 0..zip.len() {
            let mut file = zip.by_index(i).context(ManifestZIPContext)?;
            if !file.is_file() {
                continue;
            }
            let filename = match file.enclosed_name() {
                None => {
                    warn!("Skipping invalid path {}", file.name());
                    continue;
                }
                Some(f) => f,
            };

            match filename.file_name() {
                Some(f) if f.to_str() == Some("payload.bin") => {
                    if payload.is_some() {
                        return InvalidManifestContext {
                            message: "duplicate payload.bin",
                        }
                        .fail();
                    }
                    let mut buf = Vec::with_capacity(file.size() as usize);
                    file.read_to_end(&mut buf).context(ManifestFileIOContext {
                        filename: "payload.bin",
                    })?;
                    payload = Some(buf);
                }
                Some(f) if f.to_str() == Some("manifest.json") => {
                    if manifest.is_some() {
                        return InvalidManifestContext {
                            message: "duplicate manifest.json",
                        }
                        .fail();
                    }
                    let m = ModuleManifest::from_reader(file)?;
                    manifest = Some(m);
                }
                _ => (),
            };
        }

        match (manifest, payload) {
            (Some(_), None) => Err(Error::InvalidManifestError {
                message: "no payload.bin found".into(),
            }),
            (None, Some(_)) => Err(Error::InvalidManifestError {
                message: "no manifest.json found".into(),
            }),
            (None, None) => Err(Error::InvalidManifestError {
                message: "no manifest.json and no payload.bin found".into(),
            }),
            (Some(m), Some(p)) => {
                let ret = ModulePackage {
                    payload: p,
                    manifest: m,
                };
                ret.validate()?;
                Ok(ret)
            }
        }
    }

    fn identify_payload(&self) -> Result<UserModuleType> {
        let magic = read_u32_le(&mut Cursor::new(&self.payload))?;
        match magic {
            0x43534f55 => Ok(UserModuleType::Oscillator), // {'U','O','S','C'}
            0x4c454455 => Ok(UserModuleType::Delay),      // {'U','D','E','L'}
            0x444f4d55 => Ok(UserModuleType::Modulation), // {'U','M','O','D'}
            0x56455255 => Ok(UserModuleType::Reverb),     // {'U','R','E','V'}
            _ => PayloadContext {
                message: format!("Unknown magic 0x{:x}", magic),
            }
            .fail(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let m = &self.manifest;
        if m.module == UserModuleType::Unknown {
            return InvalidManifestContext {
                message: "Module type unknown",
            }
            .fail();
        }
        let payload_type = self.identify_payload()?;
        if m.module != payload_type {
            return InvalidManifestContext {
                message: format!(
                    "Module specified in manifest ({}) does not match payload type ({})",
                    m.module, payload_type
                ),
            }
            .fail();
        }
        Ok(())
    }

    fn into_repr(self) -> Result<Vec<u8>> {
        let mut ret = self.manifest.into_repr()?;

        // Payload starts at 0x400, with the length prefixed as 32-bit LE.
        ret.resize(0x400 - 4, 0);
        ret.extend((self.payload.len() as u32).to_le_bytes());
        ret.extend(self.payload);

        ret.resize(ret.len() + 132, 0); // 132 bytes padding.

        Ok(ret)
    }
}

#[repr(u8)]
enum MidiConstants {
    Sysex = 0xf0,
    EndOfSysex = 0xf7,
    Korg = 0x42,
    NonRealtime = 0x7e,
}

#[repr(u8)]
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    EnumString,
    FromRepr,
    Display,
    EnumIter,
)]
pub enum UserModuleType {
    #[strum(to_string = "Global", serialize = "global")]
    Global = 0,

    #[strum(to_string = "Modulation", serialize = "modfx", serialize = "mod")]
    Modulation = 1,

    #[strum(to_string = "Delay", serialize = "delfx", serialize = "del")]
    Delay = 2,

    #[strum(to_string = "Reverb", serialize = "revfx", serialize = "rev")]
    Reverb = 3,

    #[strum(to_string = "Oscillator", serialize = "osc")]
    Oscillator = 4,

    #[strum(disabled)]
    Unknown = 0xff,
}

impl UserModuleType {
    pub fn variants() -> Vec<UserModuleType> {
        Self::iter()
            .filter(|t| *t != Self::Global && *t != Self::Unknown)
            .collect()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserModuleInfoUnknownFields(pub u8, pub Vec<u8>);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct UserModuleInfo {
    pub module: UserModuleType,
    pub max_program_size: u32,
    pub max_load_size: u32,
    pub available_slot_count: u8,
    pub unknown: UserModuleInfoUnknownFields,
}

impl UserModuleInfo {
    fn from_repr(msg: &[u8]) -> Result<Option<Self>> {
        if msg.len() <= 1 {
            return Ok(None);
        }

        let (module, unk1) = (msg[0], msg[1]);
        let msg = undo_7bitize(&msg[2..msg.len()]);

        let mut c = Cursor::new(msg.as_ref());
        Ok(Some(UserModuleInfo {
            module: UserModuleType::from_repr(module).unwrap_or(UserModuleType::Unknown),
            max_program_size: read_u32_le(&mut c)?,
            max_load_size: read_u32_le(&mut c)?,
            available_slot_count: read_u8(&mut c)?,
            unknown: UserModuleInfoUnknownFields(unk1, remaining_slice(&c).to_vec()),
        }))
    }
}

impl Display for UserModuleInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} slots, max payload size: {}, max load size: {}",
            self.module, self.available_slot_count, self.max_program_size, self.max_load_size
        )
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, IntoStaticStr, FromRepr, Display)]
enum KorgSysexType {
    // Requests
    UserAPIVersionRequest = 0x17,
    UserModuleInfoRequest = 0x18,
    UserSlotStatusRequest = 0x19,
    UserSlotDataRequest = 0x1a,
    ClearUserSlotRequest = 0x1b,
    ClearUserModuleRequest = 0x1d,
    SwapUserDataRequest = 0x1e,

    // Generic status replies
    OperationCompleted = 0x23,
    OperationError = 0x24,
    DataFormatError = 0x26,
    UserDataSizeError = 0x27,
    UserDataCRCError = 0x28,
    UserTargetError = 0x29,
    UserAPIError = 0x2a,
    UserLoadSizeError = 0x2b,
    UserModuleError = 0x2c,
    UserSlotError = 0x2d,
    UserFormatError = 0x2e,
    UserInternalError = 0x2f,

    // Replies
    UserAPIVersionReply = 0x47,
    UserModuleInfoReply = 0x48,
    UserSlotStatusReply = 0x49,
    UserSlotData = 0x4a,
}

pub struct Device {
    _input: MidiInputConnection<()>,
    input_channel: mpsc::Receiver<Vec<u8>>,
    output: MidiOutputConnection,
}

impl Device {
    fn read(&self) -> Result<Vec<u8>> {
        self.input_channel
            .recv_timeout(READ_TIMEOUT)
            .context(DeviceReadTimeoutContext)
    }

    pub fn identify(&mut self) -> Result<DeviceIdentity> {
        let msg = &[
            MidiConstants::Sysex as u8,
            MidiConstants::NonRealtime as u8,
            0x7f, // All call
            0x06, // General information
            0x01, // Identity request
            MidiConstants::EndOfSysex as u8,
        ];
        debug!("Sending: {:x?}", msg);
        self.output.send(msg).context(DeviceCommunicationContext)?;

        let msg = self.read()?;
        DeviceIdentity::from_repr(msg.as_slice())
    }

    fn send_korg_sysex(&mut self, did: &DeviceIdentity, payload: &[u8]) -> Result<()> {
        let mut msg = Vec::with_capacity(payload.len() + 7);
        let channel_byte = 0x30 | (did.channel & 0x0f);
        msg.extend_from_slice(&[
            MidiConstants::Sysex as u8,
            MidiConstants::Korg as u8,
            channel_byte,
            0x00,
        ]);
        msg.extend(did.family_id.to_be_bytes()); // Big endian
        msg.extend_from_slice(payload);
        msg.extend_from_slice(&[MidiConstants::EndOfSysex as u8]);
        debug!("Sending: {:x?}", msg);

        self.output.send(&msg).context(DeviceCommunicationContext)?;
        Ok(())
    }

    fn recv_korg_sysex(&mut self, did: &DeviceIdentity, msgtype: KorgSysexType) -> Result<Vec<u8>> {
        let msg = self.read()?;
        let channel_byte = 0x30 | (did.channel & 0x0f);
        let msg = msg
            .strip_prefix(&[
                MidiConstants::Sysex as u8,
                MidiConstants::Korg as u8,
                channel_byte,
                0x00,
            ])
            .with_context(|| DeviceResponseParseContext {
                message: "unexpected prefix",
                offset: Some(0),
                buf: msg.clone(),
            })?
            .strip_prefix(did.family_id.to_be_bytes().as_ref()) // Big endian
            .with_context(|| DeviceResponseParseContext {
                message: "unexpected family ID",
                offset: Some(4),
                buf: msg.clone(),
            })?
            .strip_suffix(&[0xf7])
            .with_context(|| DeviceResponseParseContext {
                message: "unexpected suffix",
                offset: None,
                buf: msg.clone(),
            })?;
        match msg.strip_prefix(&[msgtype as u8]) {
            Some(ret) => Ok(Vec::from(ret)),
            None => {
                let (b, t) = if msg.is_empty() {
                    (0, "<EOF>")
                } else {
                    let b = msg[0];
                    if let Some(t) = KorgSysexType::from_repr(b) {
                        (b, t.into())
                    } else {
                        (b, "<Unknown>")
                    }
                };
                DeviceResponseParseContext {
                    message: format!(
                        "Unexpected reply: Received {} (0x{:x}), expected {} (0x{:x})",
                        t, b, msgtype, msgtype as u8
                    ),
                    offset: Some(0),
                    buf: msg,
                }
                .fail()
            }
        }
    }

    pub fn user_api_version(&mut self, did: &DeviceIdentity) -> Result<UserAPIVersion> {
        self.send_korg_sysex(did, &[KorgSysexType::UserAPIVersionRequest as u8])?;
        let p = self.recv_korg_sysex(did, KorgSysexType::UserAPIVersionReply)?;
        let r = UserAPIVersion::from_repr(p.as_slice())?;
        Ok(r)
    }

    pub fn user_slot_status(
        &mut self,
        did: &DeviceIdentity,
        module: UserModuleType,
        slot: u8,
    ) -> Result<Option<UserSlotStatus>> {
        self.send_korg_sysex(
            did,
            &[
                KorgSysexType::UserSlotStatusRequest as u8,
                module as u8,
                slot,
            ],
        )?;

        let p = self.recv_korg_sysex(did, KorgSysexType::UserSlotStatusReply)?;
        let r = UserSlotStatus::from_repr(p.as_slice())?;
        Ok(r)
    }

    pub fn user_module_info(
        &mut self,
        did: &DeviceIdentity,
        module: UserModuleType,
    ) -> Result<Option<UserModuleInfo>> {
        self.send_korg_sysex(
            did,
            &[KorgSysexType::UserModuleInfoRequest as u8, module as u8],
        )?;

        let p = self.recv_korg_sysex(did, KorgSysexType::UserModuleInfoReply)?;
        let r = UserModuleInfo::from_repr(p.as_slice())?;
        Ok(r)
    }

    pub fn upload_module(
        &mut self,
        did: &DeviceIdentity,
        module: UserModuleType,
        slot: u8,
        filename: &Path,
    ) -> Result<()> {
        let crc = Crc::<u32>::new(&CRC_32_ISO_HDLC);
        let pkg = ModulePackage::from_file(filename)?;
        info!("payload type: {:?}", pkg.identify_payload()?);
        let payload = pkg.into_repr()?;

        let crc32 = crc.checksum(payload.as_ref());
        let size = payload.len() as u32;
        info!("payload size: {:x}, crc32: {:x}", size, crc32);

        let mut payload_with_header = Vec::with_capacity(payload.len() + 8);
        payload_with_header.extend(size.to_le_bytes());
        payload_with_header.extend(crc32.to_le_bytes());
        payload_with_header.extend(payload);

        let payload_7bitized = do_7bitize(&payload_with_header);
        let mut buf = vec![KorgSysexType::UserSlotData as u8, module as u8, slot];
        buf.extend(&payload_7bitized);
        self.send_korg_sysex(did, &buf)?;
        let p = self.recv_korg_sysex(did, KorgSysexType::OperationCompleted)?;
        debug!("Upload reply: {:x?}", p);

        Ok(())
    }

    pub fn clear_user_slot(
        &mut self,
        did: &DeviceIdentity,
        module: UserModuleType,
        slot: u8,
    ) -> Result<()> {
        self.send_korg_sysex(
            did,
            &[
                KorgSysexType::ClearUserSlotRequest as u8,
                module as u8,
                slot as u8,
            ],
        )?;
        let p = self.recv_korg_sysex(did, KorgSysexType::OperationCompleted)?;
        debug!("Clear user slot reply: {:x?}", p);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate pretty_assertions;
    extern crate rand;

    use super::*;
    use pretty_assertions::assert_eq;
    use rand::Rng;

    #[test]
    fn test_version() {
        let v = Version(0xa, 0xb, 0xc);
        assert_eq!(Version::from_str("10.11-12").unwrap(), v);
        assert_eq!(v.as_repr(), [0xc, 0xb, 0xa, 0]);
    }

    #[test]
    fn test_device_identity() {
        let data = &[
            0xf0, 0x7e, 0x1, 0x6, 0x2, 0x42, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0xf7,
        ];
        assert_eq!(
            DeviceIdentity::from_repr(data).unwrap(),
            DeviceIdentity {
                channel: 0x1,
                manufacturer: 0x42,
                family_id: 0x201,
                member_id: 0x403,
                major_ver: 0x605,
                minor_ver: 0x807,
            }
        );
    }

    #[test]
    fn test_user_slot_status() {
        assert_eq!(
            UserSlotStatus::from_repr(
                vec![
                    0x4, 0x0, 0x0, 0x7f, 0x4, 0x2, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x77, 0x61, 0x76, 0x0, 0x65, 0x73,
                    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6, 0x0, 0x0, 0x0, 0x0
                ]
                .as_ref()
            )
            .unwrap(),
            Some(UserSlotStatus {
                api_version: UserAPIVersion {
                    platform: Platform::Unknown,
                    version: Version(129, 128, 128)
                },
                module: UserModuleType::Oscillator,
                slot: 0,
                developer_id: 0x80,
                program_id: 0x0,
                program_version: Version(1, 0, 0),
                program_name: "waves".to_string(),
                num_params: 6,
                module2: UserModuleType::Unknown,
                unknown: UserSlotStatusUnknownFields(0, 128, 0, vec![0, 0, 0]),
            })
        );

        assert_eq!(
            UserSlotStatus::from_repr(
                vec![
                    0x4, 0x6, 0x0, 0x0, 0x4, 0x3, 0x0, 0x1, 0x1, 0x0, 0x78, 0x78, 0x56, 0x34, 0x12,
                    0x70, 0x5e, 0x3c, 0x1a, 0x0, 0x3, 0x2, 0x1, 0x0, 0x77, 0x61, 0x76, 0x0, 0x65,
                    0x73, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3, 0x0, 0x0, 0x0,
                    0x0
                ]
                .as_ref()
            )
            .unwrap(),
            Some(UserSlotStatus {
                api_version: UserAPIVersion {
                    platform: Platform::NutektDigital,
                    version: Version(1, 1, 0)
                },
                module: UserModuleType::Oscillator,
                slot: 6,
                developer_id: 0x12345678,
                program_id: 0x9ABCDEF0,
                program_version: Version(1, 2, 3),
                program_name: "waves".to_string(),
                num_params: 3,
                module2: UserModuleType::Oscillator,
                unknown: UserSlotStatusUnknownFields(0, 0, 0, vec![0, 0, 0]),
            })
        );

        assert_eq!(
            UserSlotStatus::from_repr(vec![0x3, 0x7].as_ref()).unwrap(),
            None,
        );
    }

    #[test]
    fn test_user_module_info() {
        assert_eq!(UserModuleInfo::from_repr(vec![0].as_ref()).unwrap(), None);

        assert_eq!(
            UserModuleInfo::from_repr(
                vec![
                    0x1, 0x0, 0x1, 0x74, 0x1f, 0x0, 0x0, 0x0, 0x18, 0x0, 0x0, 0x0, 0x10, 0x0, 0x0,
                    0x0
                ]
                .as_ref()
            )
            .unwrap(),
            Some(UserModuleInfo {
                module: UserModuleType::Modulation,
                max_program_size: 8180,
                max_load_size: 6144,
                available_slot_count: 16,
                unknown: UserModuleInfoUnknownFields(0, vec![0, 0, 0])
            })
        );

        assert_eq!(
            UserModuleInfo::from_repr(
                vec![
                    0x2, 0x0, 0x1, 0x70, 0x3f, 0x0, 0x0, 0x0, 0x30, 0x0, 0x0, 0x0, 0x8, 0x0, 0x0,
                    0x0
                ]
                .as_ref()
            )
            .unwrap(),
            Some(UserModuleInfo {
                module: UserModuleType::Delay,
                max_program_size: 16368,
                max_load_size: 12288,
                available_slot_count: 8,
                unknown: UserModuleInfoUnknownFields(0, vec![0, 0, 0])
            })
        );
    }

    #[test]
    fn test_7bitize() {
        assert_eq!(
            undo_7bitize(vec![0x7f, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00].as_ref()),
            vec![0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87]
        );
        assert_eq!(
            do_7bitize(vec![0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x00].as_ref()),
            vec![0x7f, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00]
        );

        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let l: usize = rng.gen_range(20..20_000);
            let mut buf = Vec::with_capacity(l);
            buf.resize(l, 0);
            rng.try_fill(buf.as_mut_slice()).unwrap();
            let encoded = do_7bitize(buf.as_ref());
            assert_eq!(undo_7bitize(encoded.as_ref()), buf);
        }
    }
}
