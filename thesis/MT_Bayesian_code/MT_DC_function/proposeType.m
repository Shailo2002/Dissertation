function ptype = proposeType(propose, CData)
    if (propose < CData.proposal(1))
        ptype = 1;       % cell birth
    elseif(propose < CData.proposal(2))
        ptype = 2;       % cell death
    elseif (propose < CData.proposal(3))
        ptype = 3;       % cell move
    elseif (propose < CData.proposal(4))
        ptype = 4;       % change rho
    else
        ptype = 5;       % change noise level
    end
end