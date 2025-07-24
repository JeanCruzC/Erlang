def validate_erlang_inputs(forecast, aht, agents, awt):
    """Validate core Erlang parameters.

    Parameters
    ----------
    forecast : float
        Expected calls/chats in the interval.
    aht : float
        Average handling time in seconds.
    agents : int
        Number of available agents.
    awt : float
        Target answer waiting time in seconds.

    Returns
    -------
    list of str
        Error messages for parameters outside allowed bounds.
    """
    errors = []

    # Forecast validations
    if forecast <= 0:
        errors.append("El forecast debe ser mayor a 0.")
    elif forecast > 1_000_000:
        errors.append("El forecast es demasiado alto.")

    # AHT validations
    if aht <= 0:
        errors.append("El AHT debe ser mayor a 0.")
    elif aht > 3600:
        errors.append("El AHT no puede exceder 3600 segundos (1 hora).")

    # Agents validations
    if agents <= 0:
        errors.append("El número de agentes debe ser mayor a 0.")
    elif agents > 10000:
        errors.append("El número de agentes es demasiado alto.")

    # AWT validations
    if awt <= 0:
        errors.append("El AWT debe ser mayor a 0.")
    elif awt > 3600:
        errors.append("El AWT no puede exceder 3600 segundos (1 hora).")

    return errors
