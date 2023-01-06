# Rogue

A command-line client for Korg Logue devices (Prologue, Minilogue, NTS-1). It’s
a clone of
[`logue-cli`](https://github.com/korginc/logue-sdk/blob/master/tools/logue-cli/get_logue_cli_linux.sh).

## Installation

```
$ rustup update  # Debian’s rust is too old :(, use https://rustup.rs/
$ git clone https://github.com/yath/rogue
$ cd rogue
$ cargo install --path .
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
ones recognized by `logue-cli` (`modfx`, delfx`, `revxf`, `osc`), but also the
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

## License

GPL 2.0 (only).
