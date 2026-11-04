# Rogue

A command-line client for Korg Logue devices (Prologue, Minilogue, NTS-1). It’s
a clone of
[`logue-cli`](https://github.com/korginc/logue-sdk/blob/master/tools/logue-cli/get_logue_cli_linux.sh).

## Installation

```
$ rustup update  # Debian’s rust is too old :(, use https://rustup.rs/
$ cargo install --git https://github.com/yath/rogue
$ rogue probe
```

Or use the provided `Dockerfile`:

```
$ git clone https://github.com/yath/rogue
$ cd rogue
$ cargo install --path .
$ docker build -t rogue .
$ docker run --rm --device /dev/snd rogue probe
```

## Usage

Options that apply to all commands are:

* `-i`: Specify the MIDI input device, by index or by full name
* `-o`: Specify the MIDI output device, by index or by full name
* `-d`: Enable debug output

All subcommands except `check` can be restricted to a specific module type with
`-m` and potentially a slot with `-s`. Accepted module types parameters are the
ones recognized by `logue-cli` (`modfx`, `delfx`, `revxf`, `osc`), but also the
short versions `mod`, `del`, `rev` and `osc` or spelled out as `Oscillator`,
`Delay`, `Modulation` and `Reverb`.

The subcommands are:

* `probe`: Print information about available modules and slots
* `load`: Load a module unit to the device
* `check`: Validate module unit
* `clear`: Clear a module from the device

For example, to print information about installed oscillators:
```
$ rogue probe -m osc
```

Delete all reverb filters:
```
$ rogue clear -m revfx -a
```

Load a user module to the first free slot:
```
$ rogue load -u filter.ntkdigunit
```

Install oscillator into first user slot, overwriting any existing module at
that position:
```
$ rogue load -m Oscillator -s 0 -u better.prlgunit
```

See `rogue help` for more details.

## NTS-3 kaoss pad

The NTS-3 takes logue SDK 2 units (`.nts3unit`), which are plain ELF files rather than
zipped packages. It has a single module type, `genericfx` (alias `fx`), with 50 slots.

```
$ rogue probe
$ rogue load -u drone.nts3unit
$ rogue load -u drone.nts3unit -s 3
```

SysEx goes through the `NTS-3 kaoss pad kit XY/KNOB` input and `NTS-3 kaoss pad kit SOUND`
output; rogue picks those over the `MIDI IN`/`MIDI OUT` (DIN jack) ports by itself.

The protocol was worked out by capturing what KORG KONTROL Editor sends, then checked
against an NTS-3 running firmware 1.1:

* Upload: the unit is sent as `UserSlotData` (`0x4a`) messages of at most 4096 bytes:
  `F0 42 3c 00 01 72 4A <module> <slot> <chunk> <last chunk> <7-bit data> F7`. The data,
  joined across chunks, is the file size (32-bit LE), a zero word, then the file itself.
  Each chunk is 7-bit-ized separately. The device answers each chunk with
  `OperationCompleted` (`0x23`).
* Slot status: `UserSlotDataRequest` (`0x1a`) returns the unit in the same chunked format.
  An empty slot answers with a single chunk without data (`4A <module> <slot> 00 00`).
  The NTS-3 interleaves MIDI clock (`F8`) bytes with the SysEx data, which rogue strips.
* `clear` with a slot (`ClearUserSlotRequest`, `0x1b`) works as on the older devices.
  `clear -a` (`ClearUserModuleRequest`, `0x1d`) hasn't been tried.

## License

GPL 2.0 (only).
