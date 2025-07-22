# Erlang Calculator Project

This repository contains a simple Erlang calculator application structured into three main modules:

- `X`: Provides arithmetic primitives and handles integer and float calculations.
- `CHAT`: Implements a command interface for text-based interactions.
- `BL`: Supplies business logic to integrate calculations into call center, chat support, and blended workflows.

## Installation

1. Install Erlang/OTP 23 or later.
2. Clone this repository and navigate to the project directory.
3. Compile the modules:

```bash
erlc x.erl chat.erl bl.erl
```

This produces `.beam` files that can be executed with the Erlang shell.

## Quick Start

Below is a sample session that demonstrates invoking the calculator from the Erlang shell. The snippet assumes the compiled modules are in the current directory.

```erlang
1> l(x), l(chat), l(bl).
2> chat:start().
%% Type commands such as: calc 1 + 2
```

The `chat:start/0` function launches an interactive interface that parses input of the form `calc <expr>` where `<expr>` is any valid expression. The module `bl` validates input and routes the request to `x` for evaluation.

## Parameter Ranges and Edge Cases

- The `X` module accepts integers in the range `-2^31` to `2^31-1` and floating point numbers per the IEEE 754 double format.
- Division by zero returns `{error, badarg}`.
- Unrecognized commands from `CHAT` result in `{error, unknown_command}`.
- For large expressions, intermediate results may overflow; `X` returns `{error, overflow}`.

## Industry Use Cases

### Call Centers
The calculator can integrate into telephony systems so that agents can evaluate costs or convert currencies while assisting customers. The `BL` module can be extended with call center logic such as retrieving customer information or pricing data.

### Chat Support
With the `CHAT` module, support agents can perform quick calculations during live chat sessions. The interface can also be embedded in chatbots for automated help desks.

### Blended Operations
"Blended" refers to environments that combine voice and chat channels. The modular design lets organizations embed the calculator into both real-time calls and messaging. Common examples include verifying plan prices, computing discounts, or assisting with troubleshooting scripts.

## Example Module Skeletons

The repository does not yet contain code, but a minimal skeleton for `x.erl` would look like:

```erlang
-module(x).
-export([add/2, sub/2, mul/2, div/2]).

add(A, B) -> A + B.
sub(A, B) -> A - B.
mul(A, B) -> A * B.
div(_A, 0) -> {error, badarg};
div(A, B) -> A / B.
```

Further implementations of `chat.erl` and `bl.erl` follow a similar pattern, exposing functions to handle user input and business rules.


